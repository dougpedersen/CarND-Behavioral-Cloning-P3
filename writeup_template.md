#**Behavioral Cloning** 

##Writeup


**Behavioral Cloning Project**

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/Nvidia_convergence.png "Final convergence"
[image6]: ./examples/center_2017_09_13_21_49_27_763.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"
[image8]: ./examples/cropped.jpg "Cropped Image"
[image9]: ./examples/left_2017_09_13_21_49_27_763.jpg "Left camera"
[image10]: ./examples/right_2017_09_13_21_49_27_763.jpg "Right camera"

## Rubric Points
###Here I address the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1_drive.mp4 (youtube link)

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

A python data generator was used to load and augment training data as needed to reduce overall RAM usage.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The final model used was based on the [nvidia  model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars). It consists of a normalization layer, 5 convolution neural network (CNN) layers with 3x3 and 5x5 filter sizes and depths between 24 and 64, and 3 fully connected layers (model.py lines 119-128) 

The model includes RELU layers to introduce nonlinearity (code lines 119-123).

The data is clipped and normalized in the model using Keras cropping and lambda layers (code line 95-96). The nvidia whitepaper doesn't discuss what kind of normalization was done and I feel this could be a source of improved performance. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on a number of different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The final model does not contain dropout layers in order to reduce overfitting. I added them to my original model, but couldn't manage to make them help performance with this setup. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132). 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to [the comma.ai one](https://github.com/commaai/research/blob/master/train_steering_model.py) referenced in a project overview page. I thought this model might be appropriate because it had good results on the comma.ai simulator.

In order to gauge how well the model was working, I split the Udacity sample image and steering angle data (center only) into a training and validation set. I found that my first model had a low mean squared error on the training set and the validation set after 2 epochs. I ran the model on the simulator and it did well, but crashed of the bridge and fork in road. 

To attempt to fix this, I added my own data with many passes though these points at various sides of the road. This resulted in the car crashing sooner on the track.  I then added the left and right camera images and set the angle correction to +/-.09 (worked better than 0.05 and 0.15).  Performance still wasn't good on turns, so I flipped all the training data to double the training and have more turn variations.   


Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

TODO: keras export pic 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

First, I used the Udacity example data set to represent good driving behavior. Next, I recorded multiple passes over the difficult areas from different positions in the road. Here is an example image of center lane driving:

![alt text][image6]

I then used images from the left side and right side cameras so that I could guarantee more non-zero steering angle data. These images show the side angles:

![alt text][image9]
![alt text][image10]

To augment the data set, I also flipped images and angles thinking that this would give training for more turns. For example, here is an image that has then been flipped:

![alt text][image7]

This was done in the data generator. After the generation process, I had 55500 data points. I then preprocessed this data at the start of the model by clipping off the top 70 pixels and the bottom 25 pixels:

![alt text][image8]

I finally randomly shuffled the data set and put 20% of the center camera data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 6 and the convergence is shown below
![alt text][image5]

A majority of the center camera training data has a desired steering angle of 0. While more variation seems desirable, I did not attempt to eliminate any points. Augmenting with left and right camera data with a correction angle seems to help vary the desired output enough.   

I used an adam optimizer so that manually training the learning rate wasn't necessary.


##Simulation
The model drives well on track 1 and does not leave the road at all. It can make it around a number of turns on track 2, but not all of them. I didn't use any training data from track 2.

The simulator was not able to save training data on my Windows 8.1 laptop, as mentioned in the readme file. I was able to get a Windows 10 laptop to save data, after I finally realized that the simulator needs to be run from my anaconda environment.