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
