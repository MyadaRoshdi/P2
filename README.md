## Project: Build a Traffic Sign Recognition Classifier
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Goals:** The aim of this project is to create a Traffic signs classfier, using Deep machine learning techniques including Convolutional Neural Networks (CNN). The model here is trained and validated so it can classify traffic sign images using the [German Traffic Sign Dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

This is my submission for the car lane detection project, which is the 2nd project in Self-driving car Nanodegree Program. 


**Steps of this project are the following:**
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualze the features maps of the NN layer1 and 2, tested on some images.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

Also, to be able to run my submitted notebook, you will need to install the following libraries to the environment:
1. conda install opencv
2. pip install plotly

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which it is already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the [project](https://github.com/MyadaRoshdi/P2), which contains the Ipython notebook and the writeup 
```sh
git clone https://github.com/MyadaRoshdi/P2
cd P2
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Submission folders/files contents:

1. Traffic_Sign_Classifier.ipynb: This is the jupyter notebook includes all my work as code with detailed description, covering all the [rubric points](https://review.udacity.com/#!/rubrics/481/view)required in the project submission.
2. Traffic_Sign_Classifier.html: the code exported as an html file.
3. writeup.md:a writeup report either as a markdown 
4. README.md: A readme file with submission description (you'r currently opening).
5. test_dataset: Contains 14- Random Images for testing , resized to 32X32X3. 
6. un_trained_test_dataset: Contains an Image doesn't belong to any of the 43- classes and not used before in either training, validation or testing datasets.
7. LeNet5_Models: Contains the used Network architecures in my solution. 
8. 5-Softmax-per-Image: Contains the visualization of the 5- softmax values for each of the 14 images found in test_dataset.
9. Signnames.csv: Contains 2 columns shows the corresponding sign name to each class from 0 to 42.



# P2
