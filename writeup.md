# **Traffic Sign Recognition** 

## Writeup Template

### That's my report that covers my work in the [Traffic signs recognition project](https://github.com/MyadaRoshdi/P2).
1st step to use my project, download it locally:
  - git clone https://github.com/MyadaRoshdi/P2
  - cd P2
2nd step, download the dataset.
  - wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip 
  - unzip the downloaded dataset: unzip traffic-signs-data 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* Explore, summarize, visualize and shuffle the data set
* Augment modified versions of the dataset to it
* Design, train and test a model architecture
* Use the model to make predictions on new images (14- new images)
* Analyze the softmax probabilities of the new images
* Visualize the features maps of the 1st 2-layers of the Network for 2 Images, one of them never been trained for.
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Image_references/Visualization.png "Visualization"
[image2]: ./Image_references/Random_grayscale.png "Grayscaling"
[image3]: ./Image_references/Random_normalized.png "Normalizing"
[image4]: ./Image_references/label_freq_original_training.png "TrainingLabelFreq"

[image3]: ./Image_references/random_noise.jpg "Random Noise"
[image4]: ./Image_references/placeholder.png "Traffic Sign 1"
[image5]: ./Image_references/placeholder.png "Traffic Sign 2"
[image6]: ./Image_references/placeholder.png "Traffic Sign 3"
[image7]: ./Image_references/placeholder.png "Traffic Sign 4"
[image8]: ./Image_references/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### 1) Files Submitted

#### 1.1) The project submission includes all required files. Detailes about submitted files can be found in details in the **Submission folders/files contents section** in the  [README](https://github.com/MyadaRoshdi/P2/blob/master/README.md) file. and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### 2) Data Set Summary & Exploration

#### 2.1) Here, I will show the  basic summary of the original data set before any modifications.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 
    Training Set:   34799 samples
* The size of the validation set is ?
   Validation Set: 4410 samples
* The size of test set is ?
   Test Set:       12630 samples  
* The shape of a traffic sign image is ?
   Image Shape: (32, 32, 3)
* The number of unique classes/labels in the data set is ?
   43 Classes
* The percentage of Validation set out of training set is?
   Percentage of Validation Set: 12.672777953389467%

**Conclusion** As shown above the Validation is around 12.67%, which didn' give a good learning behavior, as will be shown in the next sections, After data preprocessing and Augmentation, this ratio will increase to around 25% which will dramatically affect on both Validation and Testing Accuracies.

#### 2.2) Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
* This is Sample of output classes in Training set before shuffle:

[41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41
 41 41 41 41 41 41 41 41 41 41 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31
 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31 31]
 
 **Conclusion** As shown above, the data is originally not randonly distributed so I will perform data shuffling.
 
 * This is a sample of output classes in Tranng set after Shuffle:
 
 [12 10 31 38 11  1 26 30 11 40 13  4 28  8 30  4  9  4  5 11  2 19 31  6 38
  1 17  1  1 25 12  8  5 31  4 12 31  9 35  7 12  4  7 18 31  1 14 30  2  9
 27  2 40  7  5 31  4  3 31  1 28 35 33 29 15 35  8 10 12  4 14 38  7 28 42
  7  8  7 12  8  2  4 12  8 39  1 13  1 25  6 25 35 32 35 11  3 11  8 35 13
 11 25  9 11  2 12 10 26  0 11 33 34  2 12  2 38  3 19 18 40 18  5  6 23 38
 38 14 23 16 10 38  2 34 20 13  5 12 25  8 10 38  1  9 11 10  5  1 11 16 35
 11 18 35 18  5 17  2 15 28  7  2 25 38  3  7  3  9 25  2 11  1 16 12  5  1
  8 34 23 13 13  2  2 11 38  3  5 19 11 25  0  1 21 10 18 33  8 11  7 17 24
 10 35  3 35  1 26 35 28 16 17 16 17 22  1  0 23 15 17 12 38 22 12 12 13  5
  1  2  7  2 25 13 31  1 15 14 12  3  7 11 35 34  2 27  1  7  3 16 29 11 10
 31  1  2  8  9  2  3 11 25  1  1 10 12 26 11 21 35  2 13 12 36 29 14 13 10]
 
 * This is the label frequency chart  in the Original training dataset
![Fig1: TrainingLabelFreq][./Image_references/label_freq_original_training.png]
**Conclusion** As shown in Fig1, some classes are trained better than others, which will lead as will shown below to have some errors in testing those un-suffeciently trained classes. This will be enhanced by Data Augmentation.

### 3)Design and Test a Model Architecture

#### 3.1) Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


