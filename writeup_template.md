# **Traffic Sign Recognition** 

## Writeup Template

The main objective of this project is to build a neural network to classify the German traffic signs. Used convolutional neural network in order to classify traffic signs.

** Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization/Trainingset_visualization.jpg "Visualization"
[image2]: ./visualization/TestSet_visualization.jpg "Visualization"
[image3]: ./visualization/Validation_visualization.jpg "Visualization"
[image4]: ./new-signs/ChildrenCrossing.jpg "Traffic Sign 1"
[image5]: ./new-signs/road-sign-speed-limit-30.jpg "Traffic Sign 2"
[image6]: ./new-signs/roadworks.jpg "Traffic Sign 3"
[image7]: ./new-signs/wildanimals.jpg "Traffic Sign 4"
[image8]: ./new-signs/Turnrightahead.jpg "Traffic Sign 5"



### Data Set Summary & Exploration

#### 1. I used the numpy library to calculate summary statistics of the traffic

signs data set:

The code for giving the below numbers is located in the code cells #1 in the ipython notebook.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The training set, validation set and the test set has 43 different classes representing the germna traffic signs. 
Here is an exploratory visualization of the data set for 43 classes. It is a bar chart showing 43 unique classes in X-axis and Training label frequency in y-axis.

The code for getting visualization in the code cell #3 in the ipython notebook.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. As a first step, I decided to pre-process the images 
Image process steps includes the steps of converting image to grayscale image and then normalize the grayscaled image

The given dataset has color images of dimensions 32x32 with three color channels. So preprocessing was done to all the training, test and validaiton data sets. It consissts of the following steps.

The code for this step is contained in the code cell #6 and #7 of the IPython notebook.

_**Step1:**_ Grayscale the image using cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

_**Step2:**_ Normalize the above step image using grayimage-[128.0])/128.0

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:


The difference between the original data set and the augmented data set is the following ... 


#### 2. Describing final model architecture 

The code for final model is located in code cell #10 in the ipython notebook.

My final model architecture consists of the following layers:

| Layer         		|     Description	        									| 
|:---:					|:---:															| 
| Input         		| 32x32x1 GrayScale image.   									| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 .					|
| RELU					| Activation layer.												|
| Max pooling	      	| 2x2 stride,  Inputs = 28x28x6 outputs 14x14x6. 				|
| Convolution        	| 1x1 stride, Valid padding, Inputs=14x14x6 outputs 10x10x16    |
| RELU					| Activation layer												|
| Max pooling	      	| 2x2 stride,  Inputs = 10x10x16 outputs 5x5x16 				|
| Flattern layer      	| Inputs = 5x5x16 outputs 400 									|
| Fully connected layer | Input =400, output=120       									|
| RELU					| Activation layer												|
| Fully connected layer | Input =120, output=84       									|
| RELU					| Activation layer												|
| Fully connected layer | Input =84, output=10       									|

#### 3. Training the model

To train the model, I used the following parameters.
+ Number of epochs=13. 
+ Batch Size = 128
+ Learning rate = 0.004
+ Dropout = 0.65

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

The code for calculating the accuracy of the model is in the code cell #14 in the Ipython notebook.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.937
* test set accuracy of 

This solution is based on the LeNet model architecture. This model is to be proven effective in classifying the images. At first i used LeNet model by adding dropout layer and got validation accuracy of ~0.85-0.87%. Later i removed the dropout and increased the learning rate from 0.001 to 0.004 then i got the validation accuracy ~0.93-0.94%. 

To increase the validation accuracy, i need to do the data augmentation. This will be the improvement to the current training model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
**Here are five German traffic signs that I found on the web:**

![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] 
![alt text][image8]

+ The first image children crossing sign might be difficult to classify because it contains background image on the sing symbol. 
+ The second image 30 speed limit might be difficult to classify becuase of it has complex background image.
+ The third image road works might be difficult becuase it contains multiple background images and it is complex image
+ The fourth image wild animal crossing might not be difficult to predict the image.
+ The fifth image might be difficult to classify becuase it has noizy background.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions is located in the code cells #15, #16 and #17 of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing		| Children crossing								| 
| Speed limit (30km/h) 	| Speed limit (30km/h)							|
| Road work				| Priority road     							|
| Wild animals crossing	| Wild animals crossing			 				|
| Turn right ahead		| Turn right ahead     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.  For the image1 the model is giving as general caution sign (probability as 1.0) and the image doesnt contain a General caution sign. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the #18 and #19 cell of the Ipython notebook.

For the first image, the model is preciting that this is a children crossing sign (probability of 0.49). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.49         			| Children crossing								| 
| 0.41   				| Dangerous curve to the right					|
| 0.06					| Pedestrians									|
| 0.02  				| Traffic signals    			 				|
| 7.28103e-05		    | Right-of-way at the next intersection     	|

For the second image, the model is sure that this is a Speed limit 30 sign (probability 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)							| 
| 5.65959e-18   		| Speed limit (80km/h)						 	|
| 4.54966e-21			| Speed limit (20km/h)							|
| 3.67763e-22			| End of speed limit (80km/h) 		 			|
| 8.83608e-24	        | Road work 							|

For the third image, the model is recognizing as Priority road with the probality 0.93 but the actual image is Roadworks. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.93         			| Priority road 						    	| 
| 0.04   				| No entry						 			    |
| 0.01					| Stop						    	  			|
| 0.005					| No passing 				 				    |
| 0.001			 	    | Speed limit (60km/h)						    |

For the fourth image, the model is sure that this is a Wild animals crossing (probability 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Wild animals crossing							| 
| 7.68287e-23    		| Road work							 	        |
| 8.28598e-26			| Speed limit (30km/h)							|
| 1.98254e-26			| Slippery road		 							|
| 6.99043e-30		    | No passing for vehicles over 3.5 metric tons  |

For the fifth image, the model is make sure that this is a Turn right ahead sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Turn right ahead								| 
| 9.42985e-09    		| Stop 											|
| 1.53194e-09			| Road work										|
| 1.54589e-11 			| Speed limit (60km/h)   			 			|
| 1.6983e-13		    | Keep left 									|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?