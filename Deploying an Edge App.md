# Deploying an Edge APP

## Introduction

In this lesson we'll cover:

Basics of OpenCV
Handling Input Streams in OpenCV
Processing Model Outputs for Additional Useful Information
The Basics of MQTT and their use with IoT devices
Sending statistics and video streams to a server
Performance basics
And finish up by thinking about additional model use cases, as well as end user needs

## Open CV Basics

OpenCV is an open-source library for various image processing and computer vision techniques that runs on a highly optimized C++ back-end, although it is available for use with Python and Java as well. It’s often helpful as part of your overall edge applications, whether using it’s built-in computer vision techniques or handling image processing.

Uses of OpenCV
There’s a lot of uses of OpenCV. In your case, you’ll largely focus on its ability to capture and read frames from video streams, as well as different pre-processing techniques, such as resizing an image to the expected input size of your model. It also has other pre-processing techniques like converting from one color space to another, which may help in extracting certain features from a frame. There are also plenty of computer vision techniques included, such as Canny Edge detection, which helps to extract edges from an image, and it extends even to a suite of different machine learning classifiers for tasks like face detection.

Useful OpenCV function
VideoCapture - can read in a video or image and extract a frame from it for processing
resize is used to resize a given frame
cvtColor can convert between color spaces.
You may remember from awhile back that TensorFlow models are usually trained with RGB images, while OpenCV is going to load frames as BGR. There was a technique with the Model Optimizer that would build the TensorFlow model to appropriately handle BGR. If you did not add that additional argument at the time, you could use this function to convert each image to RGB, but that’s going to add a little extra processing time.
rectangle - useful for drawing bounding boxes onto an output image
imwrite - useful for saving down a given image
See the link further down below for more tutorials on OpenCV if you want to dive deeper.

QUIZ QUESTION
Which of the following can OpenCV be used for?

Capturing a video stream

Image pre-processing (re-sizing, converting between color spaces, etc.)

Detecting features like edges or lines



Drawing or writing on an image

Further Research
OpenCV has some [pretty extensive tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html) available if you want to dive deeper into this useful computer vision library. We'll look at some of the relevant material on handling camera and video inputs next.

## Handling input streams

Being able to efficiently handle video files, image files, or webcam streams is an important part of an edge application. If I were to be running the webcam on my Macbook for instance and performing inference, a surprisingly large amount of resources get used up simply to use the webcam. That’s why it’s useful to utilize the OpenCV functions built for this - they are about as optimized for general use with input streams as you will find.

Open & Read A Video
We saw the cv2.VideoCapture function in the previous video. This function takes either a zero for webcam use, or the path to the input image or video file. That’s just the first step, though. This “capture” object must then be opened with capture.open.

Then, you can basically make a loop by checking if capture.isOpened, and you can read a frame from it with capture.read. This read function can actually return two items, a boolean and the frame. If the boolean is false, there’s no further frames to read, such as if the video is over, so you should break out of the loop

Closing the Capture
Once there are no more frames left to capture, there’s a couple of extra steps to end the process with OpenCV.

First, you’ll need to release the capture, which will allow OpenCV to release the captured file or stream
Second, you’ll likely want to use cv2.destroyAllWindows. This will make sure any additional windows, such as those used to view output frames, are closed out
Additionally, you may want to add a call to cv2.waitKey within the loop, and break the loop if your desired key is pressed. For example, if the key pressed is 27, that’s the Escape key on your keyboard - that way, you can close the stream midway through with a single button. Otherwise, you may get stuck with an open window that’s a bit difficult to close on its own.

## Exercise: Handling input streams

# Handling Input Streams

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-5de618db" class="ulab-btn--primary"></button>

It's time to really get in the think of things for running your app at the edge. Being able to
appropriately handle an input stream is a big part of having a working AI or computer vision
application. 

In your case, you will be implementing a function that can handle camera, video or webcam
data as input. While unfortunately the classroom workspace won't allow for webcam usage,
you can also try that portion of your code out on your local machine if you have a webcam
available.

As such, the tests here will focus on using a camera image or a video file. You will not need to
perform any inference on the input frames, but you will need to do a few other image
processing techniques to show you have some of the basics of OpenCV down.

Your tasks are to:

1. Implement a function that can handle camera image, video file or webcam inputs
2. Use `cv2.VideoCapture()` and open the capture stream
3. Re-size the frame to 100x100
4. Add Canny Edge Detection to the frame with min & max values of 100 and 200, respectively
5. Save down the image or video output
6. Close the stream and any windows at the end of the application

You won't be able to test a webcam input in the workspace unfortunately, but you can use
the included video and test image to test your implementations.

## Solution: Handling Input Streams

Note: There are two small changes from the code on-screen for running on Linux machines versus Mac.

On Mac, cv2.VideoWriter uses cv2.VideoWriter_fourcc('M','J','P','G') to write an .mp4 file, while Linux uses 0x00000021.
On Mac, the output with the given code on using Canny Edge Detection will run fine. However, on Linux, you'll need to use np.dstack to make a 3-channel array to write back to the out file, or else the video won't be able to be opened correctly: frame = np.dstack((frame, frame, frame))
Let's walk through each of the tasks.

Implement a function that can handle camera image, video file or webcam inputs

The main thing here is just to check the input argument passed to the command line.

This will differ by application, but in this implementation, the argument parser makes note that "CAM" is an acceptable input meaning to use the webcam. In that case, the input_stream should be set to 0, as cv2.VideoCapture() can use the system camera when set to zero.

The next is checking whether the input name is a filepath containing an image file type, such as .jpg or .png. If so, you'll just set the input_stream to that path. You should also set the flag here to note it is a single image, so you can save down the image as part of one of the later steps.

The last one is for a video file. It's mostly the same as the image, as the input_stream is the filepath passed to the input argument, but you don't need to use a flag here.

A last thing you should consider in your app here is exception handling - does your app just crash if the input is invalid or missing, or does it still log useful information to the user?

Use cv2.VideoCapture() and open the capture stream

capture = cv2.VideoCapture(input_stream)
capture.open(args.input)

while capture.isOpened():
    flag, frame = cap.read()
    if not flag:
        break
It's a bit outside of the instructions, but it's also important to check whether a key gets pressed within the while loop, to make it easier to exit.

You can use:

key_pressed = cv2.waitKey(60)
to check for a key press, and then

if key_pressed == 27:
    break
to break the loop, if needed. Key 27 is the Escape button.

Re-size the frame to 100x100

image = cv2.resize(frame, (100, 100))
Add Canny Edge Detection to the frame with min & max values of 100 and 200, respectively

Canny Edge detection is useful for detecting edges in an image, and has been a useful computer vision technique for extracting features. This was a step just so you could get a little more practice with OpenCV.

edges = cv2.Canny(image,100,200)
Display the resulting frame if it's video, or save it if it is an image

For video:

cv2.imshow('display', edges)
For a single image:

cv2.imwrite('output.jpg', edges)
Close the stream and any windows at the end of the application

Make sure to close your windows here so you don't get stuck with them on-screen.

capture.release()
cv2.destroyAllWindows()
Testing the Implementation
I can then test both an image and a video with the following:

python app.py -i blue-car.jpg
python app.py -i test_video.mp4

## Gathering Useful Information from Model Outputs


Training neural networks focuses a lot on accuracy, such as detecting the right bounding boxes and having them placed in the right spot. But what should you actually do with bounding boxes, semantic masks, classes, etc.? How would a self-driving car make a decision about where to drive based solely off the semantic classes in an image?

It’s important to get useful information from your model - information from one model could even be further used in an additional model, such as traffic data from one set of days being used to predict traffic on another set of days, such as near to a sporting event.

For the traffic example, you’d likely want to count how many bounding boxes there are, but also make sure to only count once for each vehicle until it leaves the screen. You could also consider which part of the screen they come from, and which part they exit from. Does the left turn arrow need to last longer near to a big event, as all the cars seem to be heading in that direction?

In an earlier exercise, you played around a bit with the confidence threshold of bounding box detections. That’s another way to extract useful statistics - are you making sure to throw out low confidence predictions?

Gathering Useful Information from Model Outputs Quiz
Given a model for a self-driving car capable of semantic segmentation to identify road signs, pedestrians, lane lines, cars, and other objects, what other analysis could you perform on this data?

## exercise process model outputs

# Processing Model Outputs

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-4fb9f776" class="ulab-btn--primary"></button>

Let's say you have a cat and two dogs at your house. 

If both dogs are in a room together, they are best buds, and everything is going well.

If the cat and dog #1 are in a room together, they are also good friends, and everything is fine.

However, if the cat and dog #2 are in a room together, they don't get along, and you may need
to either pull them apart, or at least play a pre-recorded message from your smart speaker
to tell them to cut it out.

In this exercise, you'll receive a video where some combination or the cat and dogs may be
in view. You also will have an IR that is able to determine which of these, if any, are on screen.

While the best model for this is likely an object detection model that can identify different
breeds, I have provided you with a very basic (and overfit) model that will return three classes,
one for one or less pets on screen, one for the bad combination of the cat and dog #2, and
one for the fine combination of the cat and dog #1. This is within the exercise directory - `model.xml`.

It is up to you to add code that will print to the terminal anytime the bad combination of the 
cat and dog #2 are detected together. **Note**: It's important to consider whether you really
want to output a warning *every single time* both pets are on-screen - is your warning helpful
if it re-starts every 30th of a second, with a video at 30 fps?

## solution 
# Processing Model Outputs - Solution

My approach in this exercise was to check if the bad combination of pets was on screen,
but also to track whether I already warned them in the current incident. Now, I might also
consider re-playing the warning after a certain time period in a single consecutive incident,
but the provided video file does not really have that long of consecutive timespans.

I also output a "timestamp" by checking how many frames had been processed so far 
at 30 fps.

The next step of this, which we'll look at shortly, is how you could actually send this 
information over the Internet, so that you could get an alert or even stream the video,
if necessary.

As we get further into the lesson and consider the costs of streaming images and/or video
to a server, another consideration here could be that you also save down the video *only*
when you run into this problem situation. You could potentially have a running 30 second loop
as well stored on the local device that is constantly refreshed, but the leading 30 seconds is
stored anytime the problematic pet combination is detected.

To run the app, I just used:

```
python app.py -m model.xml
```

Since the model was provided here in the same directory.
My approach in this exercise was to check if the bad combination of pets was on screen, but also to track whether I already warned them in the current incident. Now, I might also consider re-playing the warning after a certain time period in a single consecutive incident, but the provided video file does not really have that long of consecutive timespans. I also output a "timestamp" by checking how many frames had been processed so far at 30 fps.

Before the video loop, I added:

counter = 0
incident_flag = False
Within the loop, after a frame is read, I make sure to increment the counter: counter+=1.

I made an assess_scene function for most of the processing:

def assess_scene(result, counter, incident_flag):
    '''
    Based on the determined situation, potentially send
    a message to the pets to break it up.
    '''
    if result[0][1] == 1 and not incident_flag:
        timestamp = counter / 30
        print("Log: Incident at {:.2f} seconds.".format(timestamp))
        print("Break it up!")
        incident_flag = True
    elif result[0][1] != 1:
        incident_flag = False

    return incident_flag
And I call that within the loop right after the result is available:

incident_flag = assess_scene(result, counter, incident_flag)
Running the App
To run the app, I just used:

python app.py -m model.xml
Since the model was provided here in the same directory.

## MQTT


MQTT
MQTT stands for MQ Telemetry Transport, where the MQ came from an old IBM product line called IBM MQ for Message Queues (although MQTT itself does not use queues). That doesn’t really give many hints about its use.

MQTT is a lightweight publish/subscribe architecture that is designed for resource-constrained devices and low-bandwidth setups. It is used a lot for Internet of Things devices, or other machine-to-machine communication, and has been around since 1999. Port 1883 is reserved for use with MQTT.

Publish/Subscribe
In the publish/subscribe architecture, there is a broker, or hub, that receives messages published to it by different clients. The broker then routes the messages to any clients subscribing to those particular messages.

This is managed through the use of what are called “topics”. One client publishes to a topic, while another client subscribes to the topic. The broker handles passing the message from the publishing client on that topic to any subscribers. These clients therefore don’t need to know anything about each other, just the topic they want to publish or subscribe to.

MQTT is one example of this type of architecture, and is very lightweight. While you could publish information such as the count of bounding boxes over MQTT, you cannot publish a video frame using it. Publish/subscribe is also used with self-driving cars, such as with the Robot Operating System, or ROS for short. There, a stop light classifier may publish on one topic, with an intermediate system that determines when to brake subscribing to that topic, and then that system could publish to another topic that the actual brake system itself subscribes to.

Further Research
Visit the [main site](http://mqtt.org/) for MQTT
A [helpful post]([0](https://internetofthingsagenda.techtarget.com/definition/MQTT-MQ-Telemetry-Transport)) on more of the basics of MQTT