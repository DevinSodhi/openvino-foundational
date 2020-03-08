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

Human Pose Estimation: [human-pose-estimation-0001](https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html)
Text Detection: [text-detection-0004](http://docs.openvinotoolkit.org/latest/_models_intel_text_detection_0004_description_text_detection_0004.html)
Determining Car Type & Color: [vehicle-attributes-recognition-barrier-0039](https://docs.openvinotoolkit.org/latest/_models_intel_vehicle_attributes_recognition_barrier_0039_description_vehicle_attributes_recognition_barrier_0039.html)
Note: For ease of use, these models have been added into the /home/workspace/models directory. For example, if you need to use the Text Detection model, you could find it at:

/home/workspace/models/text_detection_0004.xml

Each link above contains the documentation for the related model. In our case, we want to focus on the Inputs section of the page, wherein important information regarding the input shape, order of the shape (such as color channel first or last), and the order of the color channels, is included.

Your task is to fill out the code in three functions within preprocess_inputs.py, one for each of the three models. We have also included a potential sample image for each of the three models, that will be used with test.py to check whether the input for each model has been adjusted as expected for proper model input.

Note that each image is currently loaded as BGR with H, W, C order in the test.py file, so any necessary preprocessing to change that should occur in your three work files. Note that BGR order is used, as the OpenCV function we use to read images loads as BGR, and not RGB.

When finished, you should be able to run the test.py file and pass all three tests.



## Preprocessing Inputs - Solution ###

### Pose Estimation ###

Let's start with `pose_estimation`, and it's [related documentation](https://docs.openvinotoolkit.org/latest/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html).

I see it is in [B, C, H, W] format, with a shape of 1x3x256x456, and an expected color order
of BGR.

Since we're loading the image with OpenCV, I know it's already in BGR format. From there, 
I need to resize the image to the desired shape, but that's going to get me 256x256x3.

```
preprocessed_image = cv2.resize(preprocessed_image, (256, 456))
```

So, I need to transpose the image, where the 3rd dimension, containing the channels,
is placed first, with the other two following.

```
preprocessed_image = preprocessed_image.transpose((2,0,1))
```

Lastly, I still need to add the `1` for the batch size at the start. I can actually just reshape
to "add" the extra dimension.

```
preprocessed_image = preprocessed_image.reshape(1,3,256,456)
```

### Text Detection

Next, let's look at `text_detection`, and it's [related documentation](http://docs.openvinotoolkit.org/latest/_models_intel_text_detection_0004_description_text_detection_0004.html).

This will actually be a very similar process to above! As such, you might actually consider
whether you could add a standard "helper" for each of these, where you could just add the
desired input shape, and perform the same transformations. Note that it does require knowing
for sure that all the steps (being in BGR, resizing, transposing, reshaping) are needed for each.

Here, the only change needed is for resizing (as well as the dimensions fed into reshape):

```
cv2.resize(preprocessed_image, (768, 1280))
```

### Car Metadata

Lastly, let's cover `car_meta`, and it's [related documentation](https://docs.openvinotoolkit.org/latest/_models_intel_vehicle_attributes_recognition_barrier_0039_description_vehicle_attributes_recognition_barrier_0039.html).

Again, all we need to change is how the image is resized, and making sure we `reshape` 
correctly:

```
cv2.resize(preprocessed_image, (72, 72))
```
Run with python test.py

## Solution ##

Using the documentation pages for each model, I ended up noticing they needed essentially the same preprocessing, outside of the height and width of the input to the network. The images coming from cv2.imread were already going to be BGR, and all the models wanted BGR inputs, so I didn't need to do anything there. However, each image was coming in as height x width x channels, and each of these networks wanted channels first, along with an extra dimension at the start for batch size.

So, for each network, the preprocessing needed to 1) re-size the image, 2) move the channels from last to first, and 3) add an extra dimension of 1 to the start. Here is the function I created for this, which I could call for each separate network:

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
Then, for each model, I can just call this function with the height and width from the documentation:

Human Pose
preprocessed_image = preprocessing(preprocessed_image, 256, 456)
Text Detection
preprocessed_image = preprocessing(preprocessed_image, 768, 1280)
Car Meta
preprocessed_image = preprocessing(preprocessed_image, 72, 72)
Testing
To test your implementation, you can just run python test.py.

## Handling Network Outputs ##

Like the computer vision model types we discussed earlier, we covered the primary outputs those networks create: classes, bounding boxes, and semantic labels.

Classification networks typically output an array with the softmax probabilities by class; the argmax of those probabilities can be matched up to an array by class for the prediction.

Bounding boxes typically come out with multiple bounding box detections per image, which each box first having a class and confidence. Low confidence detections can be ignored. From there, there are also an additional four values, two of which are an X, Y pair, while the other may be the opposite corner pair of the bounding box, or otherwise a height and width.

Semantic labels give the class for each pixel. Sometimes, these are flattened in the output, or a different size than the original image, and need to be reshaped or resized to map directly back to the input.

Quiz Information
In a network like [SSD](https://arxiv.org/pdf/1512.02325.pdf) that we discussed earlier, the output is a series of bounding boxes for potential object detections, typically also including a confidence threshold, or how confident the model is about that particular detection.

Therefore, inference performed on a given image will output an array with multiple bounding box predictions including: the class of the object, the confidence, and two corners (made of xmin, ymin, xmax, and ymax) that make up the bounding box, in that order.

Further Research
[Here](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab) is a great write-up on working with SSD and its output
This [post](https://thegradient.pub/semantic-segmentation/) gets into more of the differences in moving from models with bounding boxes to those using semantic segmentation

## Running Your First Edge App ##

![inference2](Inference_Engine_02.jpg)
You have now learned the key parts of working with a pre-trained model: obtaining the model, preprocessing inputs for it, and handling its output. In the upcoming exercise, you’ll load a pre-trained model into the Inference Engine, as well as call for functions to preprocess and handle the output in the appropriate locations, from within an edge app. We’ll still be abstracting away some of the steps of dealing with the Inference Engine API until a later lesson, but these should work similarly across different models.


## Solution: Deploy an App at the Edge ##
This was a tough one! It takes a little bit to step through this solution, as I want to give you some of my own techniques to approach this rather difficult problem first. The solution video is split into three parts - the first focuses on adding in the preprocessing and output handling calls within the app itself, and then into how I would approach implementing the Car Meta model's output handling.

### Early Steps and Car Meta Model Output Handling ###

The code for calling preprocessing and utilizing the output handling functions from within app.py is fairly straightforward:

```python
preprocessed_image = preprocessing(image, h, w)
```


Lesson 2:
Leveraging Pre-Trained Models
 1. Introduction
 2. The OpenVINO™ Toolkit
 3. Pre-Trained Models in OpenVINO™
 4. Types of Computer Vision Models
 5. Case Studies in Computer Vision
 6. Available Pre-Trained Models in OpenVINO™
 7. Exercise: Loading Pre-Trained Models
 8. Solution: Loading Pre-Trained Models
 9. Optimizations on the Pre-Trained Models
 10. Choosing the Right Model for Your App
 11. Pre-processing Inputs
 12. Exercise: Pre-processing Inputs
 13. Solution: Pre-processing Inputs
 14. Handling Network Outputs
 15. Running Your First Edge App
 16. Exercise: Deploy An App at the Edge
 17. Solution: Deploy An App at the Edge
 18. Recap
 19. Lesson Glossary
Student Hub
Chat with peers and mentors
Toggle Sidebar
Solution: Deploy An App at the Edge
Solution: Deploy an App at the Edge
This was a tough one! It takes a little bit to step through this solution, as I want to give you some of my own techniques to approach this rather difficult problem first. The solution video is split into three parts - the first focuses on adding in the preprocessing and output handling calls within the app itself, and then into how I would approach implementing the Car Meta model's output handling.

Early Steps and Car Meta Model Output Handling

The code for calling preprocessing and utilizing the output handling functions from within app.py is fairly straightforward:

preprocessed_image = preprocessing(image, h, w)
This is just feeding in the input image, along with height and width of the network, which the given inference_network.load_model function actually returned for you.

output_func = handle_output(args.t)
processed_output = output_func(output, image.shape)
This is partly based on the helper function I gave you, which can return the correct output handling function by feeding in the model type. The second line actually sends the output of inference and image shape to whichever output handling function is appropriate.

Car Meta Output Handling
Given that the two outputs for the Car Meta Model are "type" and "color", and are just the softmax probabilities by class, I wanted you to just return the np.argmax, or the index where the highest probability was determined.

def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # Get rid of unnecessary dimensions
    color = output['color'].flatten()
    car_type = output['type'].flatten()
    # TODO 1: Get the argmax of the "color" output
    color_pred = np.argmax(color)
    # TODO 2: Get the argmax of the "type" output
    type_pred = np.argmax(car_type)

    return color_pred, type_pred
Run the Car Meta Model
I have moved the models used in the exercise into a models subdirectory in the /home/workspace directory, so the path used can be a little bit shorter.

python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
For the other models, make sure to update the input image -i, model type -t, and model -m accordingly.

Pose Estimation Output Handling
Handling the car output was fairly straightforward by using np.argmax, but the outputs for the pose estimation and text detection models is a bit trickier. However, there's a lot of similar code between the two. In this second part of the solution, I'll go into detail on the pose estimation model, and then we'll finish with a quick video on handling the output of the text detection model.


Pose Estimation is more difficult, and doesn't have as nicely named outputs. I noted you just need the second one in this exercise, called 'Mconv7_stage2_L2', which is just the keypoint heatmaps, and not the associations between these keypoints. From there, I created an empty array to hold the output heatmaps once they are re-sized, as I decided to iterate through each heatmap 1 by 1 and re-size it, which can't be done in place on the original output.

def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    heatmaps = output['Mconv7_stage2_L2']
    # TODO 2: Resize the heatmap back to the size of the input
    # Create an empty array to handle the output map
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    # Iterate through and re-size each heatmap
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap
Note that the input_shape[0:2][::-1] line is taking the original image shape of HxWxC, taking just the first two (HxW), and reversing them to be WxH as cv2.resize uses.

Text Detection Model Handling
Thanks for sticking in there! The code for the text detection model is pretty similar to the pose estimation one, so let's finish things off.


Text Detection had a very similar output processing function, just using the 'model/segm_logits/add' output and only needing to resize over two "channels" of output. I likely could have extracted this out into its own output handling function that both Pose Estimation and Text Detection could have used.

def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    text_classes = output['model/segm_logits/add']
    # TODO 2: Resize this output back to the size of the input
    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])

    return out_text


    Edge Application
Applications with inference run on local hardware, sometimes without network connections, such as Internet of Things (IoT) devices, as opposed to the cloud. Less data needs to be streamed over a network connection, and real-time decisions can be made.

OpenVINO™ Toolkit
The Intel® Distribution of OpenVINO™ Toolkit enables deep learning inference at the edge by including both neural network optimizations for inference as well as hardware-based optimizations for Intel® hardware.

Pre-Trained Model
Computer Vision and/or AI models that are already trained on large datasets and available for use in your own applications. These models are often trained on datasets like ImageNet. Pre-trained models can either be used as is or used in transfer learning to further fine-tune a model. The OpenVINO™ Toolkit provides a number of pre-trained models that are already optimized for inference.

Transfer Learning
The use of a pre-trained model as a basis for further training of a neural network. Using a pre-trained model can help speed up training as the early layers of the network have feature extractors that work in a wide variety of applications, and often only late layers will need further fine-tuning for your own dataset. OpenVINO™ does not deal with transfer learning, as all training should occur prior to using the Model Optimizer.

Image Classification
A form of inference in which an object in an image is determined to be of a particular class, such as a cat vs. a dog.

Object Detection
A form of inference in which objects within an image are detected, and a bounding box is output based on where in the image the object was detected. Usually, this is combined with some form of classification to also output which class the detected object belongs to.

Semantic Segmentation
A form of inference in which objects within an image are detected and classified on a pixel-by-pixel basis, with all objects of a given class given the same label.

Instance Segmentation
Similar to semantic segmentation, this form of inference is done on a pixel-by-pixel basis, but different objects of the same class are separately identified.

SSD
A neural network combining object detection and classification, with different feature extraction layers directly feeding to the detection layer, using default bounding box sizes and shapes/

YOLO
One of the original neural networks to only take a single look at an input image, whereas earlier networks ran a classifier multiple times across a single image at different locations and scales.

Faster R-CNN
A network, expanding on R-CNN and Fast R-CNN, that integrates advances made in the earlier models by adding a Region Proposal Network on top of the Fast R-CNN model for an integrated object detection model.

MobileNet
A neural network architecture optimized for speed and size with minimal loss of inference accuracy through the use of techniques like 1x1 convolutions. As such, MobileNet is more useful in mobile applications that substantially larger and slower networks.

ResNet
A very deep neural network that made use of residual, or “skip” layers that pass information forward by a couple of layers. This helped deal with the vanishing gradient problem experienced by deeper neural networks.

Inception
A neural network making use of multiple different convolutions at each “layer” of the network, such as 1x1, 3x3 and 5x5 convolutions. The top architecture from the original paper is also known as GoogLeNet, an homage to LeNet, an early neural network used for character recognition.

Inference Precision
Precision refers to the level of detail to weights and biases in a neural network, whether in floating point precision or integer precision. Lower precision leads to lower accuracy, but with a positive trade-off for network speed and size.


