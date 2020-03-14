# Inference Engine #

## Introduction ##

In this lesson we'll cover:

Basics of the Inference Engine
Supported Devices
Feeding an Intermediate Representation to the Inference Engine
Making Inference Requests
Handling Results from the Inference Engine
Integrating the Inference Model into an App



## The inference engine ##

The Inference Engine runs the actual inference on a model. It only works with the Intermediate Representations that come from the Model Optimizer, or the Intel® Pre-Trained Models in OpenVINO™ that are already in IR format.

Where the Model Optimizer made some improvements to size and complexity of the models to improve memory and computation times, the Inference Engine provides hardware-based optimizations to get even further improvements from a model. This really empowers your application to run at the edge and use up as little of device resources as possible.

The Inference Engine has a straightforward API to allow easy integration into your edge application. The Inference Engine itself is actually built in C++ (at least for the CPU version), leading to overall faster operations; however, it is very common to utilize the built-in Python wrapper to interact with it in Python code.

QUIZ QUESTION
Which of the following best describe the Inference Engine?

It provides a library of computer vision functions and performs the inference on a model.


Great work! The Inference Engine, as the name suggests, does the real legwork of inference at the edge.


Developer Documentation
You can find the developer documentation [here](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html) for working with the Inference Engine. We’ll delve deeper into it throughout the lesson.

## Supported Devices

he supported devices for the Inference Engine are all Intel® hardware, and are a variety of such devices: CPUs, including integrated graphics processors, GPUs, FPGAs, and VPUs. You likely know what CPUs and GPUs are already, but maybe not the others.

FPGAs, or Field Programmable Gate Arrays, are able to be further configured by a customer after manufacturing. Hence the “field programmable” part of the name.

VPUs, or Vision Processing Units, are going to be like the Intel® Neural Compute Stick. They are small, but powerful devices that can be plugged into other hardware, for the specific purpose of accelerating computer vision tasks.

Differences Among Hardware
Mostly, how the Inference Engine operates on one device will be the same as other supported devices; however, you may remember me mentioning a CPU extension in the last lesson. That’s one difference, that a CPU extension can be added to support additional layers when the Inference Engine is used on a CPU.

There are also some differences among supported layers by device, which is linked to at the bottom of this page. Another important one to note is regarding when you use an Intel® Neural Compute Stick (NCS). An easy, fairly low-cost method of testing out an edge app locally, outside of your own computer is to use the NCS2 with a Raspberry Pi. The Model Optimizer is not supported directly with this combination, so you may need to create an Intermediate Representation on another system first, although there are [some instructions](https://software.intel.com/en-us/articles/model-downloader-optimizer-for-openvino-on-raspberry-pi) for one way to do so on-device. The Inference Engine itself is still supported with this combination.

QUIZ QUESTION
Which of the following Intel® hardware devices are supported for optimal performance with the OpenVINO™ Toolkit’s Inference Engine?

CPUs

GPUs

VPUs (such as the Neural Compute Stick)

FPGAs


There is a wide variety of Intel® hardware available to use with the Inference Engine! There are some additional considerations around using features like Shape Inference with OpenVINO™ that do depend on the plugins used with different hardware, which you can read about through the link below the quiz.

Further Research
Depending on your device, the different plugins do have some differences in functionality and optimal configurations. You can read more on Supported Devices [here](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_supported_plugins_Supported_Devices.html).

## Using the Inference Engine with an IR ##

IECore and IENetwork
To load an IR into the Inference Engine, you’ll mostly work with two classes in the openvino.inference_engine library (if using Python):

* IECore, which is the Python wrapper to work with the Inference Engine
* IENetwork, which is what will initially hold the network and get loaded into IECore
The next step after importing is to set a couple variables to actually use the IECore and IENetwork. In the [IECore documentation](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IECore.html), no arguments are needed to initialize. To use [IENetwork](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IENetwork.html), you need to load arguments named model and weights to initialize - the XML and Binary files that make up the model’s Intermediate Representation.

### Check Supported Layers ###
In the [IECore documentation](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IECore.html), there was another function called query_network, which takes in an IENetwork as an argument and a device name, and returns a list of layers the Inference Engine supports. You can then iterate through the layers in the IENetwork you created, and check whether they are in the supported layers list. If a layer was not supported, a CPU extension may be able to help.

The device_name argument is just a string for which device is being used - ”CPU”, ”GPU”, ”FPGA”, or ”MYRIAD” (which applies for the Neural Compute Stick).

### CPU extension ###
If layers were successfully built into an Intermediate Representation with the Model Optimizer, some may still be unsupported by default with the Inference Engine when run on a CPU. However, there is likely support for them using one of the available CPU extensions.

These do differ by operating system a bit, although they should still be in the same overall location. If you navigate to your OpenVINO™ install directory, then deployment_tools, inference_engine, lib, intel64:

On Linux, you’ll see a few CPU extension files available for AVX and SSE. That’s a bit outside of the scope of the course, but look up Advanced Vector Extensions if you want to know more there. In the classroom workspace, the SSE file will work fine.
Intel® Atom processors use SSE4, while Intel® Core processors will utilize AVX.
This is especially important to make note of when transferring a program from a Core-based laptop to an Atom-based edge device. If the incorrect extension is specified in the application, the program will crash.
AVX systems can run SSE4 libraries, but not vice-versa.
On Mac, there’s just a single CPU extension file.
You can add these directly to the IECore using their full path. After you’ve added the CPU extension, if necessary, you should re-check that all layers are now supported. If they are, it’s finally time to load the model into the IECore.

Further Research
As you get more into working with the Inference Engine in the next exercise and into the future, here are a few pages of documentation I found useful in working with it.

[IE Python API](https://docs.openvinotoolkit.org/2019_R3/ie_python_api.html)
[IE Network](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IENetwork.html)
[IE Core](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IECore.html)

## Exercise: Feed an IR to the inference engine ##

# Loading Pre-Trained Models

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-3e515cac" class="ulab-btn--primary"></button>

In this exercise, you'll work to download and load a few of the pre-trained models available 
in the OpenVINO toolkit.

First, you can navigate to the [Pre-Trained Models list](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) in a separate window or tab, as well as the page that gives all of the model names [here](https://docs.openvinotoolkit.org/latest/_models_intel_index.html).

Your task here is to download the below three pre-trained models using the Model Downloader tool, as detailed on the same page as the different model names. Note that you *do not need to download all of the available pre-trained models* - doing so would cause your workspace to crash, as the workspace will limit you to 3 GB of downloaded models.

### Task 1 - Find the Right Models
Using the [Pre-Trained Model list](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models), determine which models could accomplish the following tasks (there may be some room here in determining which model to download):
- Human Pose Estimation
- Text Detection
- Determining Car Type & Color

### Task 2 - Download the Models
Once you have determined which model best relates to the above tasks, use the Model Downloader tool to download them into the workspace for the following precision levels:
- Human Pose Estimation: All precision levels
- Text Detection: FP16 only
- Determining Car Type & Color: INT8 only

**Note**: When downloading the models in the workspace, add the `-o` argument (along with any other necessary arguments) with `/home/workspace` as the output directory. The default download directory will not allow the files to be written there within the workspace, as it is a read-only directory.

### Task 3 - Verify the Downloads
You can verify the download of these models by navigating to: `/home/workspace/intel` (if you followed the above note), and checking whether a directory was created for each of the three models, with included subdirectories for each precision, with respective `.bin` and `.xml` for each model.

**Hint**: Use the `-h` command with the Model Downloader tool if you need to check out the possible arguments to include when downloading specific models and precisions.

# Feed an IR to the Inference Engine

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-6f2a60e5" class="ulab-btn--primary"></button>

Earlier in the course, you were focused on working with the Intermediate Representation (IR)
models themselves, while mostly glossing over the use of the actual Inference Engine with
the model.

Here, you'll import the Python wrapper for the Inference Engine (IE), and practice using 
different IRs with it. You will first add each IR as an `IENetwork`, and check whether the layers 
of that network are supported by the classroom CPU.

Since the classroom workspace is using an Intel CPU, you will also need to add a CPU
extension to the `IECore`.

Once you have verified all layers are supported (when the CPU extension is added),
you will load the given model into the Inference Engine.

Note that the `.xml` file of the IR should be given as an argument when running the script.

To test your implementation, you should be able to successfully load each of the three IR
model files we have been working with throughout the course so far, which you can find in the
`/home/workspace/models` directory.

## README for solution 
# Integrate the Inference Engine - Solution

Let's step through the tasks one by one, with a potential approach for each.

> Convert a bounding box model to an IR with the Model Optimizer.

I used the SSD Mobilenet V2 architecture from TensorFlow from the earlier lesson here. Note
that the original was downloaded in a separate workspace, so I needed to download it again
and then convert it.

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

> Extract the results from the inference request

```
self.exec_network.requests[0].outputs[self.output_blob]
```

> Add code to make the requests and feed back the results within the application

```
self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
...
status = self.exec_network.requests[0].wait(-1)
```

> Add a command line argument to allow for different confidence thresholds for the model

I chose to use `-ct` as the argument name here, and added it to the existing arguments.

```
optional.add_argument("-ct", help="The confidence threshold to use with the bounding boxes", default=0.5)
```

I set a default of 0.5, so it does not need to be input by the user every time. 

> Add a command line argument to allow for different bounding box colors for the output

Similarly, I added the `-c` argument for inputting a bounding box color.
Note that in my approach, I chose to only allow "RED", "GREEN" and "BLUE", which also
impacts what I'll do in the next step; there are many possible approaches here.

```
optional.add_argument("-c", help="The color of the bounding boxes to draw; RED, GREEN or BLUE", default='BLUE')
```

> Correctly utilize the command line arguments in #3 and #4 within the application

Both of these will come into play within the `draw_boxes` function. For the first, a new line
should be added before extracting the bounding box points that check whether `box[2]`
(e.g. the probability of a given box) is above `args.ct` - assuming you have added 
`args.ct` as an argument passed to the `draw_boxes` function. If not, the box
should not be drawn. Without this, any random box will be drawn, which could be a ton of
very unlikely bounding box detections.

The second is just a small adjustment to the `cv2.rectangle` function that draws the 
bounding boxes we found to be above `args.ct`. I actually added a function to match
the different potential colors up to their RGB values first, due to how I took them in from the
command line:

```
def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']
```

I can also add the tuple returned from this function as an additional `color` argument to feed to
`draw_boxes`.

Then, the line where the bounding boxes are drawn becomes:

```
cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
```

I was able to run my app, if I was using the converted TF model from earlier (and placed in the 
current directory), using the below:

```bash
python app.py -m frozen_inference_graph.xml
```

Or, if I added additional customization with a confidence threshold of 0.6 and blue boxes:

```bash
python app.py -m frozen_inference_graph.xml -ct 0.6 -c BLUE
```

[Note that I placed my customized app actually in `app-custom.py`]

## Solution 

First, add the additional libraries (os may not be needed depending on how you get the model file names):

```python
### Load the necessary libraries

import os
from openvino.inference_engine import IENetwork, IECore


def load_to_IE(model_xml):
    ### Load the Inference Engine API
    plugin = IECore()

    ### Load IR files into their related class
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)

    ### Add a CPU extension, if applicable.
    plugin.add_extension(CPU_EXTENSION, "CPU")

    ### Get the supported layers of the network
    supported_layers = plugin.query_network(network=net, device_name="CPU")

    ### Check for any unsupported layers, and let the user
    ### know if anything is missing. Exit the program, if so.
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        print("Check whether extensions are available to add to IECore.")
        exit(1)

    ### Load the network into the Inference Engine
    plugin.load_network(net, "CPU")

    print("IR successfully loaded into Inference Engine.")

    return

```

Note that a more optimal approach here would actually check whether a CPU extension was added as an argument by the user, but to keep things simple, I hard-coded it for the exercise.

Running Your Implementation
You should make sure your implementation runs with all three pre-trained models we worked with earlier (and you are welcome to also try the models you converted in the previous lesson from TensorFlow, Caffe and ONNX, although your workspace may not have these stored). I placed these in the /home/workspace/models directory for easier use, and because the workspace will reset the /opt directory between sessions.

python feed_network.py -m /home/workspace/models/human-pose-estimation-0001.xml
You can run the other two by updating the model name in the above.

## Sending Inference Requests to the IE ## 

After you load the IENetwork into the IECore, you get back an ExecutableNetwork, which is what you will send inference requests to. There are two types of inference requests you can make: Synchronous and Asynchronous. There is an important difference between the two on whether your app sits and waits for the inference or can continue and do other tasks.

With an ExecutableNetwork, synchronous requests just use the infer function, while asynchronous requests begin with start_async, and then you can wait until the request is complete. These requests are InferRequest objects, which will hold both the input and output of the request.

We'll look a little deeper into the difference between synchronous and asynchronous on the next page.

Further Research
[Executable Network documentation](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1ExecutableNetwork.html)
[Infer Request documentation](https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1InferRequest.html)

## Asynchronous requests ##

Synchronous
Synchronous requests will wait and do nothing else until the inference response is returned, blocking the main thread. In this case, only one frame is being processed at once, and the next frame cannot be gathered until the current frame’s inference request is complete.

Asynchronous
You may have heard of asynchronous if you do front-end or networking work. In that case, you want to process things asynchronously, so in case the response for a particular item takes a long time, you don’t hold up the rest of your website or app from loading or operating appropriately.

Asynchronous, in our case, means other tasks may continue while waiting on the IE to respond. This is helpful when you want other things to still occur, so that the app is not completely frozen by the request if the response hangs for a bit.

Where the main thread was blocked in synchronous, asynchronous does not block the main thread. So, you could have a frame sent for inference, while still gathering and pre-processing the next frame. You can make use of the "wait" process to wait for the inference result to be available.

You could also use this with multiple webcams, so that the app could "grab" a new frame from one webcam while performing inference for the other.

QUIZ QUESTION
In the below examples, would an asynchronous or synchronous call make more sense?

Submit to check your answer choices!
EXAMPLE

SYNC VS. ASYNC

A network call is made to a server with an unknown latency for returning a response, and the user is otherwise able to use the app while waiting on a response.
Asynchronous

The application needs to wait for a user input before being able to process additional data.
Synchronous 

Further Research
For more on Synchronous vs. Asynchronous, check out this [helpful post](https://whatis.techtarget.com/definition/synchronous-asynchronous-API).
You can also check out the [documentation](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_Integrate_with_customer_application_new_API.html) on integrating the inference engine into an application to see the different functions calls from an Inference Request for sync (Infer) vs. async (StartAsync).
Lastly, for further practice with Asynchronous Inference Requests, you can check out [this useful demo](https://github.com/opencv/open_model_zoo/blob/master/demos/object_detection_demo_ssd_async/README.md). You’ll get a chance to practice with Synchronous and Asynchronous Requests in the upcoming exercise.


## Exercise : INference request##


# Inference Requests

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-ceb2f99a" class="ulab-btn--primary"></button>

In the previous exercise, you loaded Intermediate Representations (IRs) into the Inference
Engine. Now that we've covered some of the topics around requests, including the difference
between synchronous and asynchronous requests, you'll add additional code to make
inference requests to the Inference Engine.

Given an `ExecutableNetwork` that is the IR loaded into the Inference Engine, your task is to:

1. Perform a synchronous request
2. Start an asynchronous request given an input image frame
3. Wait for the asynchronous request to complete

Note that we'll cover handling the results of the request shortly, so you don't need to worry
about that just yet. This will get you practice with both types of requests with the Inference
Engine.

You will perform the above tasks within `inference.py`. This will take three arguments,
one for the model, one for the test image, and the last for what type of inference request
should be made.

You can use `test.py` afterward to verify your code successfully makes inference requests.


def sync_inference(exec_net, input_blob, image):
    '''
    Performs synchronous inference
    Return the result of inference
    '''
    result = exec_net.infer({input_blob: image})

    return result


    I don't actually need time.sleep() here - using the -1 with wait() is able to perform similar functionality.

Testing
You can run the test file to check your implementations using inference on multiple models.

python test.py

```python 
import argparse
import cv2
from helpers import load_to_IE, preprocessing

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the image input"
    r_desc = "The type of inference request: Async ('A') or Sync ('S')"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    parser.add_argument("-i", help=i_desc)
    parser.add_argument("-r", help=i_desc)
    args = parser.parse_args()

    return args


def async_inference(exec_net, input_blob, image):
    '''
    Performs asynchronous inference
    Returns the `exec_net`
    '''
    exec_net.start_async(request_id=0, inputs={input_blob: image})
    while True:
        status = exec_net.requests[0].wait(-1)
        if status == 0:
            break
        else:
            time.sleep(1)
    return exec_net

def sync_inference(exec_net, input_blob, image):
    '''
    Performs synchronous inference
    Return the result of inference
    '''
    result = exec_net.infer({input_blob: image})

    return result


def perform_inference(exec_net, request_type, input_image, input_shape):
    '''
    Performs inference on an input image, given an ExecutableNetwork
    '''
    # Get input image
    image = cv2.imread(input_image)
    # Extract the input shape
    n, c, h, w = input_shape
    # Preprocess it (applies for the IRs from the Pre-Trained Models lesson)
    preprocessed_image = preprocessing(image, h, w)

    # Get the input blob for the inference request
    input_blob = next(iter(exec_net.inputs))

    # Perform either synchronous or asynchronous inference
    request_type = request_type.lower()
    if request_type == 'a':
        output = async_inference(exec_net, input_blob, preprocessed_image)
    elif request_type == 's':
        output = sync_inference(exec_net, input_blob, preprocessed_image)
    else:
        print("Unknown inference request type, should be 'A' or 'S'.")
        exit(1)

    # Return the exec_net for testing purposes
    return output


def main():
    args = get_args()
    exec_net, input_shape = load_to_IE(args.m, CPU_EXTENSION)
    perform_inference(exec_net, args.r, args.i, input_shape)


if __name__ == "__main__":
    main()

```

## Handling Results

You saw at the end of the previous exercise that the inference requests are stored in a requests attribute in the ExecutableNetwork. There, we focused on the fact that the InferRequest object had a wait function for asynchronous requests.

Each InferRequest also has a few attributes - namely, inputs, outputs and latency. As the names suggest, inputs in our case would be an image frame, outputs contains the results, and latency notes the inference time of the current request, although we won’t worry about that right now.

It may be useful for you to print out exactly what the outputs attribute contains after a request is complete. For now, you can ask it for the data under the “prob” key, or sometimes output_blob [(see related documentation)](https://docs.openvinotoolkit.org/2019_R3/classInferenceEngine_1_1Blob.html), to get an array of the probabilities returned from the inference request.


QUIZ QUESTION
Which of the following code snippets could be used to extract the output from an inference request, given an ExecutableNetwork named exec_net?


exec_net.requests[request_id].outputs[output_blob]



Nice work! An ExecutableNetwork contains an InferRequest attribute by the name of requests, and feeding a given request ID key to this attribute will get the specific inference request in question.

From within this InferRequest object, it has an attribute of outputs from which you can use your output_blob to get the results of that inference request.

## Integrate into your app

In the upcoming exercise, you’ll put all your skills together, as well as adding some further customization to your app.

Further Research
There’s a ton of great potential edge applications out there for you to build. Here are some examples to hopefully get you thinking:

[Intel®’s IoT Apps Across Industries](https://www.intel.com/content/www/us/en/internet-of-things/industry-solutions.html)

[Starting Your First IoT Project](https://hackernoon.com/the-ultimate-guide-to-starting-your-first-iot-project-8b0644fbbe6d)

[OpenVINO™ on a Raspberry Pi and Intel® Neural Compute Stick](https://www.pyimagesearch.com/2019/04/08/openvino-opencv-and-movidius-ncs-on-the-raspberry-pi/)


## Exercise: Integrating into an App

# Loading Pre-Trained Models

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-3e515cac" class="ulab-btn--primary"></button>

In this exercise, you'll work to download and load a few of the pre-trained models available 
in the OpenVINO toolkit.

First, you can navigate to the [Pre-Trained Models list](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) in a separate window or tab, as well as the page that gives all of the model names [here](https://docs.openvinotoolkit.org/latest/_models_intel_index.html).

Your task here is to download the below three pre-trained models using the Model Downloader tool, as detailed on the same page as the different model names. Note that you *do not need to download all of the available pre-trained models* - doing so would cause your workspace to crash, as the workspace will limit you to 3 GB of downloaded models.

### Task 1 - Find the Right Models
Using the [Pre-Trained Model list](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models), determine which models could accomplish the following tasks (there may be some room here in determining which model to download):
- Human Pose Estimation
- Text Detection
- Determining Car Type & Color

### Task 2 - Download the Models
Once you have determined which model best relates to the above tasks, use the Model Downloader tool to download them into the workspace for the following precision levels:
- Human Pose Estimation: All precision levels
- Text Detection: FP16 only
- Determining Car Type & Color: INT8 only

**Note**: When downloading the models in the workspace, add the `-o` argument (along with any other necessary arguments) with `/home/workspace` as the output directory. The default download directory will not allow the files to be written there within the workspace, as it is a read-only directory.

### Task 3 - Verify the Downloads
You can verify the download of these models by navigating to: `/home/workspace/intel` (if you followed the above note), and checking whether a directory was created for each of the three models, with included subdirectories for each precision, with respective `.bin` and `.xml` for each model.

**Hint**: Use the `-h` command with the Model Downloader tool if you need to check out the possible arguments to include when downloading specific models and precisions.





## Integrate the Inference Engine in An Edge App

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-d44d77ce" class="ulab-btn--primary"></button>

You've come a long way from the first lesson where most of the code for working with
the OpenVINO toolkit was happening in the background. You worked with pre-trained models,
moved up to converting any trained model to an Intermediate Representation with the
Model Optimizer, and even got the model loaded into the Inference Engine and began making
inference requests.

In this final exercise of this lesson, you'll close off the OpenVINO workflow by extracting
the results of the inference request, and then integrating the Inference Engine into an existing
application. You'll still be given some of the overall application infrastructure, as more that of
will come in the next lesson, but all of that is outside of OpenVINO itself.

You will also add code allowing you to try out various confidence thresholds with the model,
as well as changing the visual look of the output, like bounding box colors.

Now, it's up to you which exact model you want to use here, although you are able to just
re-use the model you converted with TensorFlow before for an easy bounding box dectector.

Note that this application will run with a video instead of just images like we've done before.

So, your tasks are to:

1. Convert a bounding box model to an IR with the Model Optimizer.
2. Pre-process the model as necessary.
3. Use an async request to perform inference on each video frame.
4. Extract the results from the inference request.
5. Add code to make the requests and feed back the results within the application.
6. Perform any necessary post-processing steps to get the bounding boxes.
7. Add a command line argument to allow for different confidence thresholds for the model.
8. Add a command line argument to allow for different bounding box colors for the output.
9. Correctly utilize the command line arguments in #3 and #4 within the application.

When you are done, feed your model to `app.py`, and it will generate `out.mp4`, which you
can download and view. *Note that this app will take a little bit longer to run.* Also, if you need
to re-run inference, delete the `out.mp4` file first.

You only need to feed the model with `-m` before adding the customization; you should set
defaults for any additional arguments you add for the color and confidence so that the user
does not always need to specify them.

```bash
python app.py -m {your-model-path.xml}
```


## Solution 

Note: There is one small change from the code on-screen for running on Linux machines versus Mac. On Mac, cv2.VideoWriter uses cv2.VideoWriter_fourcc('M','J','P','G') to write an .mp4 file, while Linux uses 0x00000021.

Functions in inference.py
I covered the async and wait functions here as it's split out slightly differently than we saw in the last exercise.

First, it's important to note that output and input blobs were grabbed higher above when the network model is loaded:

self.input_blob = next(iter(self.network.inputs))
self.output_blob = next(iter(self.network.outputs))
From there, you can mostly use similar code to before:

    def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=0, 
            inputs={self.input_blob: image})
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[0].wait(-1)
        return status
You can grab the network output using the appropriate request with the output_blob key:

    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs[self.output_blob]
app.py
The next steps in app.py, before customization, are largely based on using the functions in inference.py:

    ### Initialize the Inference Engine
    plugin = Network()

    ### Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    ...

        ### Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### Perform inference on the frame
        plugin.async_inference(p_frame)

        ### Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            out.write(frame)
The draw_boxes function is used to extract the bounding boxes and draw them back onto the input image.

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame
Customizing app.py
Adding the customization only took a few extra steps.

Parsing the command line arguments
First, you need to add the additional command line arguments:

    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    ...

    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
The names and descriptions here, and even how you use the default values, can be up to you.

Handle the new arguments
I needed to also process these arguments a little further. This is pretty open based on your own implementation - since I took in a color string, I need to convert it to a BGR tuple for use as a OpenCV colors.

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']
I then need to call this with the related argument, as well as make sure the confidence threshold argument is a float value.

    args.c = convert_color(args.c)
    args.ct = float(args.ct)
Adding customization to draw_boxes()
The final step was to integrate these new arguments into my draw_boxes() function. I needed to make sure that the arguments are fed to the function:

frame = draw_boxes(frame, result, args, width, height)
and then I can use them where appropriate in the updated function.

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
    return frame
With everything implemented, I could run my app as such (given I re-used the previously converted TF model from the Model Optimizer lesson) if I wanted blue bounding boxes and a confidence threshold of 0.6:

python app.py -m frozen_inference_graph.xml -ct 0.6 -c BLUE

## Integrate the Inference Engine - Solution
```python

## custom solution 
import argparse
import cv2
from inference import Network

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    args = parser.parse_args()

    return args


def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
    return frame


def infer_on_video(args):
    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.ct = float(args.ct)

    ### TODO: Initialize the Inference Engine
    plugin = Network()

    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Perform inference on the frame
        plugin.async_inference(p_frame)

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            out.write(frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()




```

Let's step through the tasks one by one, with a potential approach for each.

> Convert a bounding box model to an IR with the Model Optimizer.

I used the SSD Mobilenet V2 architecture from TensorFlow from the earlier lesson here. Note
that the original was downloaded in a separate workspace, so I needed to download it again
and then convert it.

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

> Extract the results from the inference request

```
self.exec_network.requests[0].outputs[self.output_blob]
```

> Add code to make the requests and feed back the results within the application

```
self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
...
status = self.exec_network.requests[0].wait(-1)
```

> Add a command line argument to allow for different confidence thresholds for the model

I chose to use `-ct` as the argument name here, and added it to the existing arguments.

```
optional.add_argument("-ct", help="The confidence threshold to use with the bounding boxes", default=0.5)
```

I set a default of 0.5, so it does not need to be input by the user every time. 

> Add a command line argument to allow for different bounding box colors for the output

Similarly, I added the `-c` argument for inputting a bounding box color.
Note that in my approach, I chose to only allow "RED", "GREEN" and "BLUE", which also
impacts what I'll do in the next step; there are many possible approaches here.

```
optional.add_argument("-c", help="The color of the bounding boxes to draw; RED, GREEN or BLUE", default='BLUE')
```

> Correctly utilize the command line arguments in #3 and #4 within the application

Both of these will come into play within the `draw_boxes` function. For the first, a new line
should be added before extracting the bounding box points that check whether `box[2]`
(e.g. the probability of a given box) is above `args.ct` - assuming you have added 
`args.ct` as an argument passed to the `draw_boxes` function. If not, the box
should not be drawn. Without this, any random box will be drawn, which could be a ton of
very unlikely bounding box detections.

The second is just a small adjustment to the `cv2.rectangle` function that draws the 
bounding boxes we found to be above `args.ct`. I actually added a function to match
the different potential colors up to their RGB values first, due to how I took them in from the
command line:

```
def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']
```

I can also add the tuple returned from this function as an additional `color` argument to feed to
`draw_boxes`.

Then, the line where the bounding boxes are drawn becomes:

```
cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
```

I was able to run my app, if I was using the converted TF model from earlier (and placed in the 
current directory), using the below:

```bash
python app.py -m frozen_inference_graph.xml
```

Or, if I added additional customization with a confidence threshold of 0.6 and blue boxes:

```bash
python app.py -m frozen_inference_graph.xml -ct 0.6 -c BLUE
```

[Note that I placed my customized app actually in `app-custom.py`]


## BEhind the scenes of inference engine

I noted early on that the Inference Engine is built and optimized in C++, although that’s just the CPU version. There are some differences in what is actually occurring under the hood with the different devices. You are able to work with a shared API to interact with the Inference Engine, while largely being able to ignore these differences.

Why C++?
Why is the Inference Engine built in C++, at least for CPUs? In fact, many different Computer Vision and AI frameworks are built with C++, and have additional Python interfaces. OpenCV and TensorFlow, for example, are built primarily in C++, but many users interact with the libraries in Python. C++ is faster and more efficient than Python when well implemented, and it also gives the user more direct access to the items in memory and such, and they can be passed between modules more efficiently.

C++ is compiled & optimized ahead of runtime, whereas Python basically gets read line by line when a script is run. On the flip side, Python can make it easier for prototyping and fast fixes. It’s fairly common then to be using a C++ library for the actual Computer Vision techniques and inferencing, but with the application itself in Python, and interacting with the C++ library via a Python API.

Optimizations by Device
The exact optimizations differ by device with the Inference Engine. While from your end interacting with the Inference Engine is mostly the same, there’s actually separate plugins within for working with each device type.

CPUs, for instance, rely on the Intel® Math Kernel Library for Deep Neural Networks, or MKL-DNN. CPUs also have some extra work to help improve device throughput, especially for CPUs with higher numbers of cores.

GPUs utilize the Compute Library for Deep Neural Networks, or clDNN, which uses OpenCL within. Using OpenCL introduces a small overhead right when the GPU Plugin is loaded, but is only a one-time overhead cost. The GPU Plugin works best with FP16 models over FP32 models

Getting to VPU devices, like the Intel® Neural Compute Stick, there are additional costs associated with it being a USB device. It’s actually recommended to be processing four inference requests at any given time, in order to hide the costs of data transfer from the main device to the VPU.

Behind the Scenes of Inference Engine Quiz
In your own words, how does the inference engine help with deploying an AI model at the edge?
Answer:
it simplifies and speeds up the underlying models for the hardware its being deployed on.


Enter your response here, there's no right or wrong answer
Further Research
The best programming language for machine learning and deep learning is still being debated, but here’s a [great blog post](https://towardsdatascience.com/what-is-the-best-programming-language-for-machine-learning-a745c156d6b7) to give you some further background on the topic.

You can check out the [Optimization Guide](https://docs.openvinotoolkit.org/2019_R3/_docs_optimization_guide_dldt_optimization_guide.html) for more on the differences in optimization between devices.

In this lesson we covered:

Basics of the Inference Engine
Supported Devices
Feeding an Intermediate Representation to the Inference Engine
Making Inference Requests
Handling Results from the Inference Engine
Integrating the Inference Model into an App


Inference Engine
Provides a library of computer vision functions, supports calls to other computer vision libraries such as OpenCV, and performs optimized inference on Intermediate Representation models. Works with various plugins specific to different hardware to support even further optimizations.

Synchronous
Such requests wait for a given request to be fulfilled prior to continuing on to the next request.

Asynchronous
Such requests can happen simultaneously, so that the start of the next request does not need to wait on the completion of the previous.

IECore
https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IECore.html
The main Python wrapper for working with the Inference Engine. Also used to load an IENetwork, check the supported layers of a given network, as well as add any necessary CPU extensions.

IENetwork
https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1IENetwork.html
A class to hold a model loaded from an Intermediate Representation (IR). This can then be loaded into an IECore and returned as an Executable Network.

ExecutableNetwork
https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1InferRequest.html
https://docs.openvinotoolkit.org/2019_R3/classie__api_1_1ExecutableNetwork.html
An instance of a network loaded into an IECore and ready for inference. It is capable of both synchronous and asynchronous requests, and holds a tuple of InferRequest objects.

InferRequest
Individual inference requests, such as image by image, to the Inference Engine. Each of these contain their inputs as well as the outputs of the inference request once complete.