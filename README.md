# Image classifier Implementation with Convolutional Neural Networks and Keras: #

1.	#import the necessary packages
2.	from keras.models import Sequential
3.	from keras.layers.convolutional import Conv2D
4.	from keras.layers.convolutional import MaxPooling2D
5.	from keras.layers.core import Activation
6.	from keras.layers.core import Flatten
7.	from keras.layers.core import Dense
8.	from keras import backend as K
9.	
10.	class LeNet:
11.	@staticmethod	
12.	              def build(width, height, depth, classes):
13.	              # initialize the model
14.	                      model = Sequential()
15.	                      inputShape = (height, width, depth)
16.	
17.	              # if we are using "channels first", update the input shape
18.	                      if K.image_data_format() == "channels_first":
19.	                                    inputShape = (depth, height, width)


<b>Lines 2-8 </b> handle importing required Python packages. The <b>Conv2D</b> class is responsible for performing convolution. We can use the MaxPooling2D class for max-pooling operations. The Activation class applies a particular activation function.
The LeNet class is defined on Line 10 followed by the build method on Line 12 that builds the architecture. It takes 4 parameters:
width: The width of input images
height: The height of the input images
depth: The number of channels in our input images (1  for grayscale single channel images, 3  for standard RGB images)
classes: The total number of classes to recognize (in this case, two)
We define our model on Line 14. We use the Sequential class as we will be sequentially adding layers to the model.
Line 15-19 initializes our inputShape.
After initializing the model, we add layers to it:
21. # first set of CONV => RELU => POOL layers
22.  		model.add(Conv2D(20, (3, 3), padding="same"
23.		input_shape=inputShape))
24. 		model.add(Activation("relu"))
25.		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

Lines 21-25 creates the first set of CONV => RELU => POOL layers.
The CONV layer will learn 20 convolution filters, each of which are 3×3 in shape. Then we apply a ReLU activation function followed by Max-pooling with a stride of 2.

Now, we define the second set of CONV => RELU => POOL layers with 50 Conv2D layers each of 5×5.
26. 	# second set of CONV => RELU => POOL layers
27.  		model.add(Conv2D(50, (5, 5), padding="same"
28.		input_shape=inputShape))
29. 		model.add(Activation("relu"))
30.		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
Finally, we introduce a set of fully-connected layers.

31. 	# first (and only) set of FC => RELU layers
32.  		model.add(Flatten())
33.		model.add(Dense(500))
34. 		model.add(Activation("relu"))
35.		
36.	# softmax classifier
37.		model.add(Dense(classes))
38.		model.add(Activation("softmax"))
39.
40.	# return the constructed network architecture
41.		return model

Line 32 takes the output of the preceding MaxPooling2D layer and flatten it into a single vector. 
The fully-connected layer contains 500 nodes (Line 34) and they are passed through ReLU activation.
Line 38 defines another fully-connected layer, where the number of nodes is equal to the number of classes. It is then fed to softmax classifier which returns the probability for each class.
Finally, Line 41 returns our fully constructed architecture.

# Training CNN with Keras:   
1.	# import the necessary packages
2.	from keras.preprocessing.image import ImageDataGenerator
3.	from keras.optimizers import Adam
4.	from sklearn.model_selection import train_test_split
5.	from keras.preprocessing.image import img_to_array
6.	from keras.utils import to_categorical
7.	from keras.layers import Input
8.	from CancerDetection import LeNet
9.	from imutils import paths
10.	import matplotlib.pyplot as plt
11.	import numpy as np
12.	import argparse
13.	import random
14.	import cv2
15.	import os
16.	import pickle

Lines 2-16 imports necessary packages.
17.	# initialize the number of epochs to train for, initial learning rate, 
18.	# and batch size
19.	EPOCHS = 50
20.	INIT_LR = 1e-3
21.	BS = 32   

22.	# initialize the data and labels
23.	print("[INFO] loading images...")
24.	data = []
25.	labels = []

26.	# Retrieve the image paths and randomly shuffle them
27.	imagePaths = sorted(list(paths.list_images("images")))
28.	random.seed(42)
29.	random.shuffle(imagePaths)

Then we initialize data and label lists. Then we retrieve the paths to our input images and shuffle (Lines 27-29). 
30.	# loop over the input images
31.	for imagePath in imagePaths:
32.	    # load the image, pre-process it, and store it in the data list
33.	    image = cv2.imread(imagePath)
34.	    image = cv2.resize(image, (64, 64))
35.	    image = img_to_array(image)
36.	    data.append(image)
37.	
38.	    # extract the class label from the image path and update the labels list
39.	    label = imagePath.split(os.path.sep)[-2]     # Ex: images/Benign/gzl70.png
40.	    #print("label........", label)        # Malignant/Benign
41.	    label = 1 if label == "Malignant" else 0
42.	    labels.append(label)
43.	    
44.	    # scale the raw pixel intensities to the range [0, 1]
45.	data = np.array(data, dtype="float") / 255.0

The loop at Line 30 loads and resizes each image to a fixed 64×64 pixels, and appends the image array to the data list followed by extracting the class label from the imagePath on Lines 39-42. Further, on Line 45, we perform normalization to reduce the pixel values to the range [0, 1].

46.	pickle.dump(data, open("image_data.p", "wb"))

On Line 46, we pickle (compress) the preprocessed data for future re-use.

47.	dataP = pickle.load(open("image_data.p","rb"))
48.	# partition the data into training and testing splits using 75% of
49.	# the data for training and the remaining 25% for testing
50.	(trainX, testX, trainY, testY) = train_test_split(dataP, 
51.	    labels, test_size=0.25, random_state=42)


On Line 47, we load the saved pickle and perform training/testing split on the data in the ratio 3:1 (Lines 50-51). 

We also convert the labels from integers to vectors (Lines 53-54).
52.	# convert the labels from integers to vectors
53.	trainY = to_categorical(trainY, num_classes=2)
54.	testY = to_categorical(testY, num_classes=2)

55.	# construct the image generator
56.	aug = ImageDataGenerator(rotation_range=40, width_shift_range=0.1,
57.	    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, 
58.	    horizontal_flip=True, fill_mode="nearest")


Lines 56-58 construct an image generator to perform random rotations, shifts, flips, and sheers on image dataset. 

59.	# initialize the model
60.	print("[INFO] compiling model...")
61.	model = LeNet.build(width=64, height=64, depth=3, classes=2)
62.	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
63.	model.compile(loss="binary_crossentropy", optimizer=opt,
64.	    metrics=["accuracy"])
65.	
66.	# train the network
67.	print("[INFO] training network...")
68.	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
69.	    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, 
70.	    epochs=EPOCHS, verbose=1)
71.	
72.	# save the model to disk
73.	print("[INFO] serializing network...")
74.	model.save("cancer_not_cancer.model")


Now, we build our model using LeNet with Adam optimizer (Lines 61-63) and use binary_crossentropy as the loss function. We then train the network by supplying train/test data, number of epochs and other parameters (Lines 67-70). Lines (72-74) allow us to save our trained model to disk for future re-use.

75.	# plot the training loss and accuracy
76.	plt.style.use("ggplot")
77.	plt.figure()
78.	N = EPOCHS
79.	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
80.	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
81.	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
82.	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
83.	plt.title("Training Loss and Accuracy on Cancer/Not Cancer")
84.	plt.xlabel("Epoch #")
85.	plt.ylabel("Loss/Accuracy")
86.	plt.legend(loc="lower left")
87.	plt.savefig("plot.png")


##########################

As evident from the figure above, the network is trained for 50 epochs and achieved high accuracy (80.68% testing accuracy) and low loss. Epoch-wise results are shown below. 

555555555555555

# Accuracy of CNN image classifier:
88.	print("[INFO] loading network...")
89.	model = load_model("cancer_not_cancer.model")
90.	scores = model.evaluate(testX, testY)
91.	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

Lines (88-89) reloads the saved model from the disk. Lines (90-91) evaluates the model’s accuracy by supplying unseen data. The model has returned an accuracy of 80.69% and the probability that the below test image belongs to Benign class is calculated as 89.85%.

fdjhdfhdfjdshfjdsfdsfsdkfd






