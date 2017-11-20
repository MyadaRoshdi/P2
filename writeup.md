# **Traffic Sign Recognition** 

## Writeup Report

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

[image1]: ./Images-references/label_freq_original_training.png "Labels Visualization"
[image2]: ./Images-references/Visualization.png "Random Images Visualization"
[image3]: ./Images-references/Random_grayscale.png "Random grayscaling"
[image4]: ./Images-references/Random_rotated_brighened.png "Random modified"
[image5]: ./LeNet5_Models/2-stage-ConvNet-architecture.png "Modified LeNet5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### 1) Files Submitted

#### 1.1) The project submission includes all required files. Detailes about submitted files can be found in details in the **Submission folders/files contents section** in the  [README](https://github.com/MyadaRoshdi/P2/blob/master/README.md) file. and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### 2) Data Set Summary & Exploration

#### 2.1) Here, I will show the  basic summary of the original data set before any modifications.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* **The size of training set is ?** 
   Training Set:   34799 samples
* **The size of the validation set is ?**
   Validation Set: 4410 samples
* **The size of test set is ?**
   Test Set:       12630 samples  
* **The shape of a traffic sign image is ?**
   Image Shape: (32, 32, 3)
* **The number of unique classes/labels in the data set is ?**
   43 Classes
* **The percentage of Validation set out of training set is?**
   Percentage of Validation Set: 12.672777953389467%

**Conclusion:** As shown above the Validation is around 12.67%, which didn' give a good learning behavior, as will be shown in the next sections, After data preprocessing and Augmentation, this ratio will increase to around 25% which will dramatically affect on both Validation and Testing Accuracies.

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
 
 **Conclusion:** As shown above, the data is originally not randonly distributed so I will perform data shuffling.
 
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
 
![alt text][image1]

**Conclusion:** As shown in above chart, some classes are trained better than others, which will lead as will shown below to have some errors in testing those un-suffeciently trained classes. This will be enhanced by Data Augmentation.

* This is Visualization of some Random images and their correspondong sign-names
 
![alt text][image2]



### 3)Design and Test a Model Architecture

#### 3.1) **Preprocessing:** here, I will describe how I preprocessed the image data.
* a) grayscaling: Here I converted and image to grayscale, so now shape changed from (32X32X3) to (32X32X1), as shown below.

![alt text][image3]

* b) Normalizing: Here I used the formula  _(pixel - 128)/ 128_

Datasets Mean values BEFORE Normalization are:
 
Training set : 82.6775890369964
 
Validation set: 83.55642737563775 

Testing set : 82.14846036120183 

Datasets Mean values AFTER Normalization are:

Training set : -0.35408133564846533
 
Validation set: -0.3472154111278302 

Testing set : -0.3582151534281105 


**I decided to generate additional data because when I experimited on just the supported data, the Validation accuracty never exceeded 95.6%, but once I augmneted more data and used Validation dataset of around 25% from training, the Validation testing increased to 99%.

I used the input data processing techniques as suggested by the Lecun paper, now I have a jittered dataset by adding 2 transformed versions of randomly selected sets of the original training set, which are :

1) Random Image Rotation between -15 and 15 degrees

2) Random Brighness

The following figure shows an image after applying rotation and brighness.

![alt text][image4]

**NOTE:** In the notebook, you can find another method for modifying the data which is Affine transfomation, but empirically I found that removing this step enhancing the Validation accuracy.
 
Now after Augmeting more data to the original data, then split 25% of them as Validation data, we get the followng datasets to train and validate the network over them.

New X_train size: 111356

X_validation size: 27840

Percentage of Validation Set: 25.000898020762243%


#### 3.2) **Model Architecture** In the first set of experiments, I used the original LeNet5 Architecture (described in details in the [notebook](https://github.com/MyadaRoshdi/P2/blob/master/Traffic_Sign_Classifier.ipynb)), but the Validation accuracy didn't exceed 95% , then I tried the Using Suggested Architecture in [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), which achieved a significant improvment.

My final model consisted of the following layers as shown in the figure below:

![alt text][image5]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 	|			
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16     |
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 1x1x400    |
| RELU                  |                                               |
| Flatten Conv2  	  	| Flatten layer2 o/p, i/p = 5x5x16. o/p = 400	|	
| Flatten Conv3  	  	| Flatten layer3 o/p, i/p = 1x1x400. o/p = 400	|
| Concatenate			| Concatenate layer2 & layer3. so now o/p = 800 |					
| Fully connected		| Fully Connected. Input = 800. Output = 43.    |
| Softmax				|        									|
|						|												|
|						|												|
 


#### 3.3) **Model Training** All the data used here are fine tuned by expermenting.

To train the model, I used the following values:
 
 Type of optimizer: AdamOptimizer
 
 The batch size: 150
 
 number of epochs: 50

 learning rate: 0.0005
 
 dropout: 0.5
 
 **NOTE** In the first set of experiments, I used the dropout after the last layer in the network, I tried varying the number from 0.1 and 0.9, then when I tried to remove it, the Validation accuracy enhanced from 98.7% to 99.9%. I concluded that usng dropout cannot give a benefit when the traning set is not big enough. 

** Validation Accuracy per epoch is: **

Training...

EPOCH 1 ...
Validation Accuracy = 0.918

EPOCH 2 ...
Validation Accuracy = 0.953

EPOCH 3 ...
Validation Accuracy = 0.976

EPOCH 4 ...
Validation Accuracy = 0.983

EPOCH 5 ...
Validation Accuracy = 0.986

EPOCH 6 ...
Validation Accuracy = 0.990

EPOCH 7 ...
Validation Accuracy = 0.992

EPOCH 8 ...
Validation Accuracy = 0.990

EPOCH 9 ...
Validation Accuracy = 0.992

EPOCH 10 ...
Validation Accuracy = 0.997

EPOCH 11 ...
Validation Accuracy = 0.994

EPOCH 12 ...
Validation Accuracy = 0.994

EPOCH 13 ...
Validation Accuracy = 0.996

EPOCH 14 ...
Validation Accuracy = 0.994

EPOCH 15 ...
Validation Accuracy = 0.997

EPOCH 16 ...
Validation Accuracy = 0.993

EPOCH 17 ...
Validation Accuracy = 0.998

EPOCH 18 ...
Validation Accuracy = 0.997

EPOCH 19 ...
Validation Accuracy = 0.996

EPOCH 20 ...
Validation Accuracy = 0.999

EPOCH 21 ...
Validation Accuracy = 0.999

EPOCH 22 ...
Validation Accuracy = 0.996

EPOCH 23 ...
Validation Accuracy = 0.999

EPOCH 24 ...
Validation Accuracy = 0.999

EPOCH 25 ...
Validation Accuracy = 0.995

EPOCH 26 ...
Validation Accuracy = 0.999

EPOCH 27 ...
Validation Accuracy = 0.999

EPOCH 28 ...
Validation Accuracy = 0.999

EPOCH 29 ...
Validation Accuracy = 0.999

EPOCH 30 ...
Validation Accuracy = 0.999

EPOCH 31 ...
Validation Accuracy = 0.999

EPOCH 32 ...
Validation Accuracy = 0.998

EPOCH 33 ...
Validation Accuracy = 0.998

EPOCH 34 ...
Validation Accuracy = 0.999

EPOCH 35 ...
Validation Accuracy = 0.999

EPOCH 36 ...
Validation Accuracy = 0.999

EPOCH 37 ...
Validation Accuracy = 0.999

EPOCH 38 ...
Validation Accuracy = 0.998

EPOCH 39 ...
Validation Accuracy = 0.999

EPOCH 40 ...
Validation Accuracy = 0.999

EPOCH 41 ...
Validation Accuracy = 0.999

EPOCH 42 ...
Validation Accuracy = 0.991

EPOCH 43 ...
Validation Accuracy = 0.999

EPOCH 44 ...
Validation Accuracy = 0.999

EPOCH 45 ...
Validation Accuracy = 0.999

EPOCH 46 ...
Validation Accuracy = 0.999

EPOCH 47 ...
Validation Accuracy = 1.000

EPOCH 48 ...
Validation Accuracy = 0.998

EPOCH 49 ...
Validation Accuracy = 0.998

EPOCH 50 ...
Validation Accuracy = 0.999

Model saved

#### 3.4)**Solution Approach** Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**My final model results were:**
* **training set accuracy of ?**
* **validation set accuracy of ?** 
* **test set accuracy of ?**

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


