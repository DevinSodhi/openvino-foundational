# Leveraging Pre-Trained Models # 

## The OpenVINO Toolkit (intel distro) ##

The OpenVINO™ Toolkit’s name comes from “Open Visual Inferencing and Neural Network Optimization”. It is largely focused around optimizing neural network inference, and is open source.

It is developed by Intel®, and helps support fast inference across Intel® CPUs, GPUs, FPGAs and Neural Compute Stick with a common API. OpenVINO™ can take models built with multiple different frameworks, like TensorFlow or Caffe, and use its Model Optimizer to optimize for inference. This optimized model can then be used with the Inference Engine, which helps speed inference on the related hardware. It also has a wide variety of Pre-Trained Models already put through Model Optimizer.

By optimizing for model speed and size, OpenVINO™ enables running at the edge. This does not mean an increase in inference accuracy - this needs to be done in training beforehand. The smaller, quicker models OpenVINO™ generates, along with the hardware optimizations it provides, are great for lower resource applications. For example, an IoT device does not have the benefit of multiple GPUs and unlimited memory space to run its apps.

### QUIZ QUESTION ###
The Intel® Distribution of OpenVINO™ Toolkit is:
An open source library useful for edge deployment due to its performance maximizations and pre-trained models



## Pre-Trained Models in OpenVINO ##

In general, pre-trained models refer to models where training has already occurred, and often have high, or even cutting-edge accuracy. Using pre-trained models avoids the need for large-scale data collection and long, costly training. Given knowledge of how to preprocess the inputs and handle the outputs of the network, you can plug these directly into your own app.

In OpenVINO™, Pre-Trained Models refer specifically to the Model Zoo, in which the Free Model Set contains pre-trained models already converted using the Model Optimizer. These models can be used directly with the Inference Engine.

![image of pre trained models](Inference_Engine_01.jpg)

Further Research
We’ll come back to the various pre-trained models available with the OpenVINO™ Toolkit shortly, but you can get a headstart by checking out the documentation [here](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models).

## Types of Computer Vision Models ##

We covered three types of computer vision models in the video: Classification, Detection, and Segmentation.

Classification determines a given “class” that an image, or an object in an image, belongs to, from a simple yes/no to thousands of classes. These usually have some sort of “probability” by class, so that the highest probability is the determined class, but you can also see the top 5 predictions as well.

Detection gets into determining that objects appear at different places in an image, and oftentimes draws bounding boxes around the detected objects. It also usually has some form of classification that determines the class of an object in a given bounding box. The bounding boxes have a confidence threshold so you can throw out low-confidence detections.

Segmentation classifies sections of an image by classifying each and every pixel. These networks are often post-processed in some way to avoid phantom classes here and there. Within segmentation are the subsets of semantic segmentation and instance segmentation - the first wherein all instances of a class are considered as one, while the second actually consider separates instances of a class as separate objects.

### QUIZ QUESTION ###
Match the types of Computer Vision models below to their descriptions.

TYPE OF MODEL

Determines what an object in a given image is, although not where in the image it is located.
Answer: Classification

Determines the location of an object in an image on a pixel-by-pixel basis.
Answer: Segmentation

Determines the location of an object using some type of markers like a bounding box around the area the object is in.
Answer: Detection

More details can be found in this [article](https://medium.com/analytics-vidhya/image-classification-vs-object-detection-vs-image-segmentation-f36db85fe81)

Note there are other types of models like pose estimation and text recognition. You can generate content with gaanz as well, but those are not in this scope.


## Case Studies in Computer Vision ##



We focused on SSD, ResNet and MobileNet in the video. 
* [SSD](https://arxiv.org/abs/1512.02325) (Single shot multibox) is an object detection network that combined classification with object detection through the use of default bounding boxes at different network levels. 
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf) utilized residual layers to “skip” over sections of layers, helping to avoid the vanishing gradient problem with very deep neural networks. 
* MobileNet utilized layers like 1x1 convolutions to help cut down on computational complexity and network size, leading to fast inference without substantial decrease in accuracy.

One additional note here on the ResNet architecture - the paper itself actually theorizes that very deep neural networks have convergence issues due to exponentially lower convergence rates, as opposed to just the vanishing gradient problem. The vanishing gradient problem is also thought to be helped by the use of normalization of inputs to each different layer, which is not specific to ResNet. The ResNet architecture itself, at multiple different numbers of layers, was shown to converge faster during training than a “plain” network without the residual layers.

Common Neural architectures: 
Some of the pre trained models are build with these architectures:

*  detector - 
* 

### QUESTION 1 OF 2 ###
The Single Shot Multibox Detector (SSD) model:

Performed classifications on different convolutional layer feature maps using default bounding boxes

### QUESTION 2 OF 2 ### 
The “residual learning” achieved in the ResNet model architecture is achieved by:

Using “skip” layers that pass information forward by a couple of layers

More neural Networks: 
* [SSD](https://arxiv.org/abs/1512.02325)
* [YOLO](https://arxiv.org/abs/1506.02640)
* [Faster RCNN](https://arxiv.org/abs/1506.01497)
* [MobileNet](https://arxiv.org/abs/1704.04861)
* [ResNet](https://arxiv.org/abs/1512.03385)
* [Inception](https://arxiv.org/pdf/1409.4842.pdf)


## Available Pre-Trained Models in OpenVINO ##

Most of the Pre-Trained Models supplied by OpenVINO™ fall into either face detection, human detection, or vehicle-related detection. There is also a model around detecting text, and more!

Models in the Public Model Set must still be run through the Model Optimizer, but have their original models available for further training and fine-tuning. The Free Model Set are already converted to Intermediate Representation format, and do not have the original model available. These can be easily obtained with the Model Downloader tool provided in the files installed with OpenVINO™.

The SSD and MobileNet architectures we discussed previously are often the main part of the architecture used for many of these models.



### QUIZ QUESTION ###
Which of the below models are available in OpenVINO™ as pre-trained models?

Text Detection
Pose Detection
Roadside Segmentation
Pedestrian Detection

You can check out the full list of pre-trained models available in the Intel® Distribution of OpenVINO™ [here](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models). As we get into the Model Optimizer in the next lesson, you’ll find it’s quite easy to take pre-trained models available from other sources and use them with OpenVINO™ as well.


Models can be downloaded via a "model downloader", a python file that lets you download files. YOu can see this on the page in the documentation. The current path might look like : 
* (<OENVINO_INTSALL_DIR>/deployment_tools/open_model_zoo/tools/downloader) 
* you can use the -h argument with the file to see all the options you can have with the download. this is a hint for the upcoming exercises

* when to use which?
  * SSD : enhanced model - face detection
  * MobileNET : standard model - face detection
  * SSD + Mobilenet: pedestrian and vehicle detection


## Loading Pre-Trained Models ##

Loading Pre-Trained Models
Make sure to click the button below before you get started to source the correct environment.

SOURCE ENV

In this exercise, you'll work to download and load a few of the pre-trained models available in the OpenVINO toolkit.

First, you can navigate to the [Pre-Trained](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) Models list in a separate window or tab, as well as the page that gives all of the model names [here](https://docs.openvinotoolkit.org/latest/_models_intel_index.html).

Your task here is to download the below three pre-trained models using the Model Downloader tool, as detailed on the same page as the different model names. Note that you do not need to download all of the available pre-trained models - doing so would cause your workspace to crash, as the workspace will limit you to 3 GB of downloaded models.

Task 1 - Find the Right Models
Using the Pre-Trained Model list, determine which models could accomplish the following tasks (there may be some room here in determining which model to download):

Human Pose Estimation
Text Detection
Determining Car Type & Color
Task 2 - Download the Models
Once you have determined which model best relates to the above tasks, use the Model Downloader tool to download them into the workspace for the following precision levels:

Human Pose Estimation: All precision levels
Text Detection: FP16 only
Determining Car Type & Color: INT8 only
Note: When downloading the models in the workspace, add the -o argument (along with any other necessary arguments) with /home/workspace as the output directory. The default download directory will not allow the files to be written there within the workspace, as it is a read-only directory.

Task 3 - Verify the Downloads
You can verify the download of these models by navigating to: /home/workspace/intel (if you followed the above note), and checking whether a directory was created for each of the three models, with included subdirectories for each precision, with respective .bin and .xml for each model.

Hint: Use the -h command with the Model Downloader tool if you need to check out the possible arguments to include when downloading specific models and precisions.

### How to do it ###

the tricky thing is that you need to track down the folder where intel vino is installed

* /opt/intel/openvino/deployment_tools/tools/model_downloader# 

Then you need to fo to the above mentioned websites
* https://docs.openvinotoolkit.org/latest/_models_intel_index.html
* https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models

and get the names of the models.

The Three tasks we need to download are: 
* Human Pose Estimation
* Text Detection    FP16 ONLY
* Determining Car Type and Color. int8 only


The corresponding names seem to be: 
* human-pose-estimation-0001

using the -h as suggested, we do commands like: 
* /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name human-pose-estimation-0001
* /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name vehicle-attributes-recognition-barrier-0039 --precisions INT8 -o /home/workspace
* /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name text-detection-0003 --precisions FP16 

The Key takeaways are that we want to use the python command in the correct way. We want to use the downloader helper to understand these commands. We want to be able to get  --precision right and also --o can feed output location. Finally, we might want to use the name --name-of-model as described on the models intel docs

* in each of the models that you've downloaded, they should have listed precisions, and deeper in a binary and an xml file


## Choosing Models ##
I chose the following models for the three tasks:

Human Pose Estimation: human-pose-estimation-0001
Text Detection: text-detection-0004
Determining Car Type & Color: vehicle-attributes-recognition-barrier-0039
Downloading Models
To navigate to the directory containing the Model Downloader:

cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
Within there, you'll notice a downloader.py file, and can use the -h argument with it to see available arguments. For this exercise, --name for model name, and --precisions, used when only certain precisions are desired, are the important arguments. Note that running downloader.py without these will download all available pre-trained models, which will be multiple gigabytes. You can do this on your local machine, if desired, but the workspace will not allow you to store that much information.

Note: In the classroom workspace, you will not be able to write to the /opt/intel directory, so you should also use the -o argument to specify your output directory as /home/workspace (which will download into a created intel folder therein).

### Downloading Human Pose Model ###
sudo ./downloader.py --name human-pose-estimation-0001 -o /home/workspace
Downloading Text Detection Model
sudo ./downloader.py --name text-detection-0004 --precisions FP16 -o /home/workspace
Downloading Car Metadata Model
sudo ./downloader.py --name vehicle-attributes-recognition-barrier-0039 --precisions INT8 -o /home/workspace
Verifying Downloads
The downloader itself will tell you the directories these get saved into, but to verify yourself, first start in the /home/workspace directory (or the same directory as the Model Downloader if on your local machine without the -o argument). From there, you can cd intel, and then you should see three directories - one for each downloaded model. Within those directories, there should be separate subdirectories for the precisions that were downloaded, and then .xml and .bin files within those subdirectories, that make up the model.


## Optimizations on the Pre-Trained Models ##

In the exercise, you dealt with different precisions of the different models. Precisions are related to floating point values - less precision means less memory used by the model, and less compute resources. However, there are some trade-offs with accuracy when using lower precision. There is also fusion, where multiple layers can be fused into a single operation. These are achieved through the Model Optimizer in OpenVINO™, although the Pre-Trained Models have already been run through that process. We’ll return to these optimization techniques in the next lesson.


## Choosing the Right Model for Your App ##
Make sure to test out different models for your application, comparing and contrasting their use cases and performance for your desired task. Remember that a little bit of extra processing may yield even better results, but needs to be implemented efficiently.

This goes both ways - you should try out different models for a single use case, but you should also consider how a given model can be applied to multiple use cases. For example, being able to track human poses could help in physical therapy applications to assess and track progress of limb movement range over the course of treatment.

QUIZ QUESTION

QUIZ QUESTION
Below are a few existing pre-trained models, as well as some different applications they might be used in. In your best judgement, match these models to which application would best be fit for their use.

Submit to check your answer choices!
POTENTIAL APPLICATION

MODEL

Traffic Light Optimization
* Detect People , Vehicles and bikes
Assess Traffic Levels in Retail Aisles
* pedestrian detection
Delivery Robot
* identify roadside objects=
Monitor Form When Working Out
* human pose estimation

## Pre-processing Inputs ##

The pre-processing needed for a network will vary, but usually this is something you can check out in any related documentation, including in the OpenVINO™ Toolkit documentation. It can even matter what library you use to load an image or frame - OpenCV, which we’ll use to read and handle images in this course, reads them in the BGR format, which may not match the RGB images some networks may have used to train with.

Outside of channel order, you also need to consider image size, and the order of the image data, such as whether the color channels come first or last in the dimensions. Certain models may require a certain normalization of the images for input, such as pixel values between 0 and 1, although some networks also do this as their first layer.

In OpenCV, you can use cv2.imread to read in images in BGR format, and cv2.resize to resize them. The images will be similar to a numpy array, so you can also use array functions like .transpose and .reshape on them as well, which are useful for switching the array dimension order.

## Exercise: Pre-processing Inputs ##


Preprocessing Inputs
Make sure to click the button below before you get started to source the correct environment.

SOURCE ENV

Now that we have a few pre-trained models downloaded, it's time to preprocess the inputs to match what each of the models expects as their input. We'll use the same models as before as a basis for determining the preprocessing necessary for each input file.

As a reminder, our three models are:

Human Pose Estimation: human-pose-estimation-0001
Text Detection: text-detection-0004
Determining Car Type & Color: vehicle-attributes-recognition-barrier-0039
Note: For ease of use, these models have been added into the /home/workspace/models directory. For example, if you need to use the Text Detection model, you could find it at:


/home/workspace/models/text_detection_0004.xml
Each link above contains the documentation for the related model. In our case, we want to focus on the Inputs section of the page, wherein important information regarding the input shape, order of the shape (such as color channel first or last), and the order of the color channels, is included.

Your task is to fill out the code in three functions within preprocess_inputs.py, one for each of the three models. We have also included a potential sample image for each of the three models, that will be used with test.py to check whether the input for each model has been adjusted as expected for proper model input.

Note that each image is currently loaded as BGR with H, W, C order in the test.py file, so any necessary preprocessing to change that should occur in your three work files. Note that BGR order is used, as the OpenCV function we use to read images loads as BGR, and not RGB.

When finished, you should be able to run the test.py file and pass all three tests.

