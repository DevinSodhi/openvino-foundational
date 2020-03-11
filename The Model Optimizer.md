# Model optimizer #


In this lesson we'll cover:

Basics of the Model Optimizer
Different Optimization Techniques and their impact on model performance
Supported Frameworks in the Intel® Distribution of OpenVINO™ Toolkit
Converting from models in those frameworks to Intermediate Representations
And a bit on Custom Layers

## The Model Optimizer

The Model Optimizer helps convert models in multiple different frameworks to an Intermediate Representation, which is used with the Inference Engine. If a model is not one of the pre-converted models in the Pre-Trained Models OpenVINO™ provides, it is a required step to move onto the Inference Engine.

As part of the process, it can perform various optimizations that can help shrink the model size and help make it faster, although this will not give the model higher inference accuracy. In fact, there will be some loss of accuracy as a result of potential changes like lower precision. However, these losses in accuracy are minimized.

Local Configuration
Configuring the Model Optimizer is pretty straight forward for your local machine, given that you already have OpenVINO™ installed. You can navigate to your OpenVINO™ install directory first, which is usually /opt/intel/openvino. Then, head to /deployment_tools/model_optimizer/install_prerequisites, and run the install_prerequisites.sh script therein.

QUIZ QUESTION
Which of the following best describe the Model Optimizer?

It converts a model for use with the Inference Engine, including improvements to size and speed.

Developer Documentation
You can find the developer documentation [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) for working with the Model Optimizer. We’ll delve deeper into it throughout the lesson.

## Optimization Techniques

Here, I mostly focused on three optimization techniques: quantization, freezing and fusion. Note that at the end of the video when I mention hardware optimizations, those are done by the Inference Engine (which we’ll cover in the next lesson), not the Model Optimizer.

Quantization
Quantization is related to the topic of precision I mentioned before, or how many bits are used to represent the weights and biases of the model. During training, having these very accurate numbers can be helpful, but it’s often the case in inference that the precision can be reduced without substantial loss of accuracy. Quantization is the process of reducing the precision of a model.

With the OpenVINO™ Toolkit, models usually default to FP32, or 32-bit floating point values, while FP16 and INT8, for 16-bit floating point and 8-bit integer values, are also available (INT8 is only currently available in the Pre-Trained Models; the Model Optimizer does not currently support that level of precision). FP16 and INT8 will lose some accuracy, but the model will be smaller in memory and compute times faster. Therefore, quantization is a common method used for running models at the edge.

Freezing
Freezing in this context is used for TensorFlow models. Freezing TensorFlow models will remove certain operations and metadata only needed for training, such as those related to backpropagation. Freezing a TensorFlow model is usually a good idea whether before performing direct inference or converting with the Model Optimizer.

Fusion
Fusion relates to combining multiple layer operations into a single operation. For example, a batch normalization layer, activation layer, and convolutional layer could be combined into a single operation. This can be particularly useful for GPU inference, where the separate operations may occur on separate GPU kernels, while a fused operation occurs on one kernel, thereby incurring less overhead in switching from one kernel to the next.

QUIZ QUESTION
Match the types of optimization for inference below to their descriptions.

Submit to check your answer choices!
DESCRIPTION

OPTIMIZATION TYPE

Reduces precision of weights and biases, thereby reducing compute time and size with some loss of accuracy.

Ans: Quantization

On a layer basis is used for fine-tuning a neural network; in TensorFlow this removes metadata only needed for training.

Ans: Freezing

Combining certain operations together into one operation and needing less computational overhead.

Ans: Fusion 


Further Research
If you’d like to learn more about quantization, check out this [helpful post](https://nervanasystems.github.io/distiller/quantization.html).
You can find out more about optimizations performed by the Model Optimizer in the OpenVINO™ Toolkit [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Model_Optimization_Techniques.html).


## Supported Frameworks ##

The supported frameworks with the OpenVINO™ Toolkit are:

Caffe
TensorFlow
MXNet
ONNX (which can support PyTorch and Apple ML models through another conversion step)
Kaldi
These are all open source, just like the OpenVINO™ Toolkit. Caffe is originally from UC Berkeley, TensorFlow is from Google Brain, MXNet is from Apache Software, ONNX is combined effort of Facebook and Microsoft, and Kaldi was originally an individual’s effort. Most of these are fairly multi-purpose frameworks, while Kaldi is primarily focused on speech recognition data.

There are some differences in how exactly to handle these, although most differences are handled under the hood of the OpenVINO™ Toolkit. For example, TensorFlow has some different steps for certain models, or frozen vs. unfrozen models. However, most of the functionality is shared across all of the supported frameworks.

QUIZ QUESTION
Which of these frameworks are supported by the OpenVINO™ Toolkit?

Caffe

TensorFlow

MXNet

ONNX

Kaldi

Further Research
In case you aren’t familiar with any of these frameworks, feel free to check out the sites for each below:

[Caffe](https://caffe.berkeleyvision.org/)
[TensorFlow](https://www.tensorflow.org/)
[MXNet](https://mxnet.apache.org/)
[ONNX](https://onnx.ai/)
[Kaldi](https://kaldi-asr.org/doc/dnn.html)



## Intermediate Representations ##


Intermediate Representations (IRs) are the OpenVINO™ Toolkit’s standard structure and naming for neural network architectures. A Conv2D layer in TensorFlow, Convolution layer in Caffe, or Conv layer in ONNX are all converted into a Convolution layer in an IR.

The IR is able to be loaded directly into the Inference Engine, and is actually made of two output files from the Model Optimizer: an XML file and a binary file. The XML file holds the model architecture and other important metadata, while the binary file holds weights and biases in a binary format. You need both of these files in order to run inference Any desired optimizations will have occurred while this is generated by the Model Optimizer, such as changes to precision. You can generate certain precisions with the --data_type argument, which is usually FP32 by default.

QUIZ QUESTION
The Intermediate Representation is a model where specific layers of supported deep learning frameworks are replaced with layers in the “dialect” of the Inference Engine.

True



Further Research

You can find the main developer documentation on converting models in the OpenVINO™ Toolkit [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Converting_Model.html). We’ll cover how to do so with TensorFlow, Caffe and ONNX (useful for PyTorch) over the next several pages.

You can find the documentation on different layer names when converted to an IR [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html).
Finally, you can find more in-depth data on each of the Intermediate Representation layers themselves [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_IRLayersCatalogSpec.html).

## Using the Model Optimizer with TensorFlow Models ##

Once the Model Optimizer is configured, the next thing to do with a TensorFlow model is to determine whether to use a frozen or unfrozen model. You can either freeze your model, which I would suggest, or use the separate instructions in the documentation to convert a non-frozen model. Some models in TensorFlow may already be frozen for you, so you can skip this step.

From there, you can feed the model into the Model Optimizer, and get your Intermediate Representation. However, there may be a few items specific to TensorFlow for that stage, which you’ll need to feed into the Model Optimizer before it can create an IR for use with the Inference Engine.

TensorFlow models can vary for what additional steps are needed by model type, being unfrozen or frozen, or being from the TensorFlow Detection Model Zoo. Unfrozen models usually need the --mean_values and --scale parameters fed to the Model Optimizer, while the frozen models from the Object Detection Model Zoo don’t need those parameters. However, the frozen models will need TensorFlow-specific parameters like --tensorflow_use_custom_operations_config and --tensorflow_object_detection_api_pipeline_config. Also, --reverse_input_channels is usually needed, as TF model zoo models are trained on RGB images, while OpenCV usually loads as BGR. Certain models, like YOLO, DeepSpeech, and more, have their own separate pages.

TensorFlow Object Detection Model Zoo
The models in the TensorFlow Object Detection Model Zoo can be used to even further extend the pre-trained models available to you. These are in TensorFlow format, so they will need to be fed to the Model Optimizer to get an IR. The models are just focused on object detection with bounding boxes, but there are plenty of different model architectures available.

Further Research
The developer documentation for Converting TensorFlow Models can be found [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html). You’ll work through this process in the next exercise.
TensorFlow also has additional models available in the [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). By converting these over to Intermediate Representations, you can expand even further on the pre-trained models available to you.

* First run the configuration steps for the model optimizer.
* Freeze the TF model if your model is not frozen
  * OR use the instructions to convert a non frozen model.
* Convert the TF model with teh model optimizer to an optimized IR
  * Test the model in the IR format using the inference engine.


##  Exercise Convert a TensorFlow Model

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-663e2c8b" class="ulab-btn--primary"></button>

In this exercise, you'll convert a TensorFlow Model from the Object Detection Model Zoo
into an Intermediate Representation using the Model Optimizer.

As noted in the related [documentation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html), 
there is a difference in method when using a frozen graph vs. an unfrozen graph. Since
freezing a graph is a TensorFlow-based function and not one specific to OpenVINO itself,
in this exercise, you will only need to work with a frozen graph. However, I encourage you to
try to freeze and load an unfrozen model on your own as well.

For this exercise, first download the SSD MobileNet V2 COCO model from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz). Use the `tar -xvf` 
command with the downloaded file to unpack it.

From there, find the **Convert a TensorFlow\* Model** header in the documentation, and
feed in the downloaded SSD MobileNet V2 COCO model's `.pb` file. 

If the conversion is successful, the terminal should let you know that it generated an IR model.
The locations of the `.xml` and `.bin` files, as well as execution time of the Model Optimizer,
will also be output.

**Note**: Converting the TF model will take a little over one minute in the workspace.

### Hints & Troubleshooting

Make sure to pay attention to the note in this section regarding the 
`--reverse_input_channels` argument. 
If you are unsure about this argument, you can read more [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html#when_to_reverse_input_channels).

There is additional documentation specific to converting models from TensorFlow's Object
Detection Zoo [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html).
You will likely need both the `--tensorflow_use_custom_operations_config` and
`--tensorflow_object_detection_api_pipeline_config` arguments fed with their 
related files.

Here's what I entered to convert the SSD MobileNet V2 model from TensorFlow:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

This is pretty long! I would suggest considering setting a [path environment variable](https://help.ubuntu.com/community/EnvironmentVariables) for the Model Optimizer if you are working locally on a Linux-based machine. You could do something like this:

export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer

And then when you need to use it, you can utilize it with $MOD_OPT/mo.py instead of entering the full long path each time. In this case, that would also help shorten the path to the ssd_v2_support.json file used.

## Using the Model Optimizer with Caffe Models ##

The process for converting a Caffe model is fairly similar to the TensorFlow one, although there’s nothing about freezing the model this time around, since that’s a TensorFlow concept. Caffe does have some differences in the set of supported model architectures. Additionally, Caffe models need to feed both the .caffemodel file, as well as a .prototxt file, into the Model Optimizer. If they have the same name, only the model needs to be directly input as an argument, while if the .prototxt file has a different name than the model, it should be fed in with --input_proto as well.

Further Research
The developer documentation for Converting Caffe Models can be found [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html). You’ll work through this process in the next exercise.

## Exercise: Convert a Caffee Model

Here's what I entered to convert the Squeezenet V1.1 model from Caffe:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt


### Loading Pre-Trained Models

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

### Convert a Caffe Model

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-d0a57724" class="ulab-btn--primary"></button>

In this exercise, you'll convert a Caffe Model into an Intermediate Representation using the 
Model Optimizer. You can find the related documentation [here](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html).

For this exercise, first download the SqueezeNet V1.1 model by cloning [this repository](https://github.com/DeepScale/SqueezeNet). 

Follow the documentation above and feed in the Caffe model to the Model Optimizer.

If the conversion is successful, the terminal should let you know that it generated an IR model.
The locations of the `.xml` and `.bin` files, as well as execution time of the Model Optimizer,
will also be output.

### Hints & Troubleshooting

You will need to specify `--input_proto` if the `.prototxt` file is not named the same as the model.

There is an important note in the documentation after the section **Supported Topologies** 
regarding Caffe models trained on ImageNet. If you notice poor performance in inference, you
may need to specify mean and scale values in your arguments.

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt
```


Here's what I entered to convert the Squeezenet V1.1 model from Caffe:


python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt


## Using the Model Optimizer with ONNX Models ##

The process for converting an ONNX model is again quite similar to the previous two, although ONNX does not have any ONNX-specific arguments to the Model Optimizer. So, you’ll only have the general arguments for items like changing the precision.

Additionally, if you are working with PyTorch or Apple ML models, they need to be converted to ONNX format first, which is done outside of the OpenVINO™ Toolkit. See the link further down on this page if you are interested in doing so.

Further Research
The developer documentation for Converting ONNX Models can be found [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html). You’ll work through this process in the next exercise.
ONNX also has additional models available in the [ONNX Model Zoo](https://github.com/onnx/models). By converting these over to Intermediate Representations, you can expand even further on the pre-trained models available to you.
PyTorch to ONNX
If you are interested in converting a PyTorch model using ONNX for use with the OpenVINO™ Toolkit, check out this [link](https://michhar.github.io/convert-pytorch-onnx/) for the steps to do so. From there, you can follow the steps for ONNX models to get an Intermediate Representation.

## Exercise: Conert an ONNX Model.


# Convert an ONNX Model

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-0bd71d51" class="ulab-btn--primary"></button>

### Exercise Instructions

In this exercise, you'll convert an ONNX Model into an Intermediate Representation using the 
Model Optimizer. You can find the related documentation [here](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html).

For this exercise, first download the bvlc_alexnet model from [here](https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz). Use the `tar -xvf` command with the downloaded file to unpack it.

Follow the documentation above and feed in the ONNX model to the Model Optimizer.

If the conversion is successful, the terminal should let you know that it generated an IR model.
The locations of the `.xml` and `.bin` files, as well as execution time of the Model Optimizer,
will also be output.

### PyTorch models

Note that we will only cover converting directly from an ONNX model here. If you are interested
in converting a PyTorch model using ONNX for use with OpenVINO, check out this [link](https://michhar.github.io/convert-pytorch-onnx/) for the steps to do so. From there, you can follow the steps in the rest
of this exercise once you have an ONNX model.


Here's what I entered to convert the AlexNet model from ONNX:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx

## Cutting Parts of a model

Cutting a model is mostly applicable for TensorFlow models. As we saw earlier in converting these models, they sometimes have some extra complexities. Some common reasons for cutting are:

The model has pre- or post-processing parts that don’t translate to existing Inference Engine layers.
The model has a training part that is convenient to be kept in the model, but is not used during inference.
The model is too complex with many unsupported operations, so the complete model cannot be converted in one shot.
The model is one of the supported SSD models. In this case, you need to cut a post-processing part off.
There could be a problem with model conversion in the Model Optimizer or with inference in the Inference Engine. To localize the issue, cutting the model could help to find the problem
There’s two main command line arguments to use for cutting a model with the Model Optimizer, named intuitively as --input and --output, where they are used to feed in the layer names that should be either the new entry or exit points of the model.

Developer Documentation
You guessed it - [here’s](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Cutting_Model.html) the developer documentation for cutting a model.

## Supported Layers ##

Earlier, we saw some of the supported layers when looking at the names when converting from a supported framework to an IR. While that list is useful for one-offs, you probably don’t want to check whether each and every layer in your model is supported. You can also just see when you run the Model Optimizer what will convert.

What happens when a layer isn’t supported by the Model Optimizer? One potential solution is the use of custom layers, which we’ll go into more shortly. Another solution is actually running the given unsupported layer in its original framework. For example, you could potentially use TensorFlow to load and process the inputs and outputs for a specific layer you built in that framework, if it isn’t supported with the Model Optimizer. Lastly, there are also unsupported layers for certain hardware, that you may run into when working with the Inference Engine. In this case, there are sometimes extensions available that can add support. We’ll discuss that approach more in the next lesson.

QUIZ QUESTION

Any layer created in one of the supported frameworks is able to be directly converted by the Model Optimizer.

False

Correct! While just about every layer you’d likely be using in your own neural network is supported with the Model Optimizer, sometimes you’ll need to make use of Custom Layers, which we’ll cover next.

Supported Layers List
Check out the full list of supported layers [here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html).

## Custom Layers

Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit, although you shouldn’t have to use it very often, if at all, due to all of the supported layers. However, it’s useful to know a little about its existence and how to use it if the need arises.

The [list of supported layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) from earlier very directly relates to whether a given layer is a custom layer. Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.

To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

You’ll get a chance to practice this in the next exercise. Again, as this is an advanced topic, we won’t delve too much deeper here, but feel free to check out the linked documentation if you want to know more.

Further Research
You’ll get a chance to get hands on with Custom Layers next, but feel free to check out the [developer documentation](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html) in the meantime.

If you’re interested in the option to use TensorFlow to operate on a given unsupported layer, you should also make sure to read the [documentation here](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Offloading_Sub_Graph_Inference.html).

## exercise custom layers

# Custom Layers

Make sure to click the button below before you get started to source the correct environment.

<button id="ulab-button-c7cfa177" class="ulab-btn--primary"></button>

This exercise is adapted from [this repository](https://github.com/david-drew/OpenVINO-Custom-Layers).

Note that the classroom workspace is running OpenVINO 2019.r3, while this exercise was
originally created for 2019.r2. This exercise will work appropriately in the workspace, but there
may be some other differences you need to account for if you use a custom layer yourself.

The below steps will walk you through the full walkthrough of creating a custom layer; as such,
there is not a related solution video. Note that custom layers is an advanced topic, and one
that is not expected to be used often (if at all) in most use cases of the OpenVINO toolkit. This
exercise is meant to introduce you to the concept, but you won't need to use it again in the 
rest of this course.

## Example Custom Layer: The Hyperbolic Cosine (cosh) Function

We will follow the steps involved for implementing a custom layer using the simple 
hyperbolic cosine (cosh) function. The cosh function is mathematically calculated as:

```
cosh(x) = (e^x + e^-x) / 2
```

As a function that calculates a value for the given value x, the cosh function is very simple 
when compared to most custom layers. Though the cosh function may not represent a "real"  custom layer, it serves the purpose of this tutorial as an example for working through the steps 
for implementing a custom layer.

Move to the next page to continue.
## Build the Model

First, export the below paths to shorten some of what you need to enter later:

```
export CLWS=/home/workspace/cl_tutorial
export CLT=$CLWS/OpenVINO-Custom-Layers
```

Then run the following to create the TensorFlow model including the `cosh` layer.

```
mkdir $CLWS/tf_model
python $CLT/create_tf_model/build_cosh_model.py $CLWS/tf_model
```

You should receive a message similar to:

```
Model saved in path: /tf_model/model.ckpt
```

## Creating the *`cosh`* Custom Layer

### Generate the Extension Template Files Using the Model Extension Generator

We will use the Model Extension Generator tool to automatically create templates for all the 
extensions needed by the Model Optimizer to convert and the Inference Engine to execute 
the custom layer.  The extension template files will be partially replaced by Python and C++ 
code to implement the functionality of `cosh` as needed by the different tools.  To create 
the four extensions for the `cosh` custom layer, we run the Model Extension Generator 
with the following options:

- `--mo-tf-ext` = Generate a template for a Model Optimizer TensorFlow extractor
- `--mo-op` = Generate a template for a Model Optimizer custom layer operation
- `--ie-cpu-ext` = Generate a template for an Inference Engine CPU extension
- `--ie-gpu-ext` = Generate a template for an Inference Engine GPU extension 
- `--output_dir` = set the output directory.  Here we are using `$CLWS/cl_cosh` as the target directory to store the output from the Model Extension Generator.

To create the four extension templates for the `cosh` custom layer, given we are in the `$CLWS`
directory, we run the command:

```
mkdir cl_cosh
```

```bash
python /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir=$CLWS/cl_cosh
```

The Model Extension Generator will start in interactive mode and prompt us with questions 
about the custom layer to be generated.  Use the text between the `[]`'s to answer each 
of the Model Extension Generator questions as follows:

```
Enter layer name: 
[cosh]

Do you want to automatically parse all parameters from the model file? (y/n)
...
[n]

Enter all parameters in the following format:
...
Enter 'q' when finished:
[q]

Do you want to change any answer (y/n) ? Default 'no'
[n]

Do you want to use the layer name as the operation name? (y/n)
[y]

Does your operation change shape? (y/n)  
[n]

Do you want to change any answer (y/n) ? Default 'no'
[n]
```

When complete, the output text will appear similar to:
```
Stub file for TensorFlow Model Optimizer extractor is in /home/<user>/cl_tutorial/cl_cosh/user_mo_extensions/front/tf folder
Stub file for the Model Optimizer operation is in /home/<user>/cl_tutorial/cl_cosh/user_mo_extensions/ops folder
Stub files for the Inference Engine CPU extension are in /home/<user>/cl_tutorial/cl_cosh/user_ie_extensions/cpu folder
Stub files for the Inference Engine GPU extension are in /home/<user>/cl_tutorial/cl_cosh/user_ie_extensions/gpu folder
```

Template files (containing source code stubs) that may need to be edited have just been 
created in the following locations:

- TensorFlow Model Optimizer extractor extension: 
  - `$CLWS/cl_cosh/user_mo_extensions/front/tf/`
  - `cosh_ext.py`
- Model Optimizer operation extension:
  - `$CLWS/cl_cosh/user_mo_extensions/ops`
  - `cosh.py`
- Inference Engine CPU extension:
  - `$CLWS/cl_cosh/user_ie_extensions/cpu`
  - `ext_cosh.cpp`
  - `CMakeLists.txt`
- Inference Engine GPU extension:
  - `$CLWS/cl_cosh/user_ie_extensions/gpu`
  - `cosh_kernel.cl`
  - `cosh_kernel.xml`

Instructions on editing the template files are provided in later parts of this tutorial.  
For reference, or to copy to make the changes quicker, pre-edited template files are provided 
by the tutorial in the `$CLT` directory.

Move to the next page to continue.

## Using Model Optimizer to Generate IR Files Containing the Custom Layer 

We will now use the generated extractor and operation extensions with the Model Optimizer 
to generate the model IR files needed by the Inference Engine.  The steps covered are:

1. Edit the extractor extension template file (already done - we will review it here)
2. Edit the operation extension template file (already done - we will review it here)
3. Generate the Model IR Files

### Edit the Extractor Extension Template File

For the `cosh` custom layer, the generated extractor extension does not need to be modified 
because the layer parameters are used without modification.  Below is a walkthrough of 
the Python code for the extractor extension that appears in the file 
`$CLWS/cl_cosh/user_mo_extensions/front/tf/cosh_ext.py`.
1. Using the text editor, open the extractor extension source file `$CLWS/cl_cosh/user_mo_extensions/front/tf/cosh_ext.py`.
2. The class is defined with the unique name `coshFrontExtractor` that inherits from the base extractor `FrontExtractorOp` class.  The class variable `op` is set to the name of the layer operation and `enabled` is set to tell the Model Optimizer to use (`True`) or exclude (`False`) the layer during processing.

    ```python
    class coshFrontExtractor(FrontExtractorOp):
        op = 'cosh' 
        enabled = True
    ```

3. The `extract` function is overridden to allow modifications while extracting parameters from layers within the input model.

    ```python
    @staticmethod
    def extract(node):
    ```

4. The layer parameters are extracted from the input model and stored in `param`.  This is where the layer parameters in `param` may be retrieved and used as needed.  For the `cosh` custom layer, the `op` attribute is simply set to the name of the operation extension used.

    ```python
    proto_layer = node.pb
    param = proto_layer.attr
    # extracting parameters from TensorFlow layer and prepare them for IR
    attrs = {
        'op': __class__.op
    }
    ```

5. The attributes for the specific node are updated. This is where we can modify or create attributes in `attrs` before updating `node` with the results and the `enabled` class variable is returned.

    ```python
    # update the attributes of the node
    Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)
    
    return __class__.enabled
    ```

### Edit the Operation Extension Template File

For the `cosh` custom layer, the generated operation extension does not need to be modified 
because the shape (i.e., dimensions) of the layer output is the same as the input shape.  
Below is a walkthrough of the Python code for the operation extension that appears in 
the file  `$CLWS/cl_cosh/user_mo_extensions/ops/cosh.py`.

1. Using the text editor, open the operation extension source file `$CLWS/cl_cosh/user_mo_extensions/ops/cosh.py` 
2. The class is defined with the unique name `coshOp` that inherits from the base operation `Op` class.  The class variable `op` is set to `'cosh'`, the name of the layer operation.

    ```python
    class coshOp(Op):
    op = 'cosh'
    ```

3. The `coshOp` class initializer `__init__` function will be called for each layer created.  The initializer must initialize the super class `Op` by passing the `graph` and `attrs` arguments along with a dictionary of the mandatory properties for the `cosh` operation layer that define the type (`type`), operation (`op`), and inference function (`infer`).  This is where any other initialization needed by the `coshOP` operation can be specified.

    ```python
    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            infer=coshOp.infer            
        )
    super().__init__(graph, mandatory_props, attrs)
    ```

4. The `infer` function is defined to provide the Model Optimizer information on a layer, specifically returning the shape of the layer output for each node.  Here, the layer output shape is the same as the input and the value of the helper function `copy_shape_infer(node)` is returned.

    ```python
    @staticmethod
    def infer(node: Node):
        # ==========================================================
        # You should add your shape calculation implementation here
        # If a layer input shape is different to the output one
        # it means that it changes shape and you need to implement
        # it on your own. Otherwise, use copy_shape_infer(node).
        # ==========================================================
        return copy_shape_infer(node)
    ```

### Generate the Model IR Files

With the extensions now complete, we use the Model Optimizer to convert and optimize 
the example TensorFlow model into IR files that will run inference using the Inference Engine.  
To create the IR files, we run the Model Optimizer for TensorFlow `mo_tf.py` with 
the following options:

- `--input_meta_graph model.ckpt.meta`
  - Specifies the model input file.  

- `--batch 1`
  - Explicitly sets the batch size to 1 because the example model has an input dimension of "-1".
  - TensorFlow allows "-1" as a variable indicating "to be filled in later", however the Model Optimizer requires explicit information for the optimization process.  

- `--output "ModCosh/Activation_8/softmax_output"`
  - The full name of the final output layer of the model.

- `--extensions $CLWS/cl_cosh/user_mo_extensions`
  - Location of the extractor and operation extensions for the custom layer to be used by the Model Optimizer during model extraction and optimization. 

- `--output_dir $CLWS/cl_ext_cosh`
  - Location to write the output IR files.

To create the model IR files that will include the `cosh` custom layer, we run the commands:

```bash
cd $CLWS/tf_model
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCosh/Activation_8/softmax_output" --extensions $CLWS/cl_cosh/user_mo_extensions --output_dir $CLWS/cl_ext_cosh
```

The output will appear similar to:

```
[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/<user>/cl_tutorial/cl_ext_cosh/model.ckpt.xml
[ SUCCESS ] BIN file: /home/<user>/cl_tutorial/cl_ext_cosh/model.ckpt.bin
[ SUCCESS ] Total execution time: x.xx seconds.
```

Move to the next page to continue.## Inference Engine Custom Layer Implementation for the Intel® CPU

We will now use the generated CPU extension with the Inference Engine to execute 
the custom layer on the CPU.  The steps are:

1. Edit the CPU extension template files.
2. Compile the CPU extension library.
3. Execute the Model with the custom layer.

You *will* need to make the changes in this section to the related files.

Note that the classroom workspace only has an Intel CPU available, so we will not perform
the necessary steps for GPU usage with the Inference Engine.

### Edit the CPU Extension Template Files

The generated CPU extension includes the template file `ext_cosh.cpp` that must be edited 
to fill-in the functionality of the `cosh` custom layer for execution by the Inference Engine.  
We also need to edit the `CMakeLists.txt` file to add any header file or library dependencies 
required to compile the CPU extension.  In the next sections, we will walk through and edit 
these files.

#### Edit `ext_cosh.cpp`

We will now edit the `ext_cosh.cpp` by walking through the code and making the necessary 
changes for the `cosh` custom layer along the way.

1. Using the text editor, open the CPU extension source file `$CLWS/cl_cosh/user_ie_extensions/cpu/ext_cosh.cpp`.

2. To implement the `cosh` function to efficiently execute in parallel, the code will use the parallel processing supported by the Inference Engine through the use of the Intel® Threading Building Blocks library.  To use the library, at the top we must include the header [`ie_parallel.hpp`](https://docs.openvinotoolkit.org/2019_R3.1/ie__parallel_8hpp.html) file by adding the `#include` line as shown below.

    Before:

    ```cpp
    #include "ext_base.hpp"
    #include <cmath>
    ```

    After:

    ```cpp
    #include "ext_base.hpp"
    #include "ie_parallel.hpp"
    #include <cmath>
    ```

3. The class `coshImp` implements the `cosh` custom layer and inherits from the extension layer base class `ExtLayerBase`.

    ```cpp
    class coshImpl: public ExtLayerBase {
        public:
    ```

4. The `coshImpl` constructor is passed the `layer` object that it is associated with to provide access to any layer parameters that may be needed when implementing the specific instance of the custom layer.

    ```cpp
    explicit coshImpl(const CNNLayer* layer) {
      try {
        ...
    ```

5. The `coshImpl` constructor configures the input and output data layout for the custom layer by calling `addConfig()`.  In the template file, the line is commented-out and we will replace it to indicate that `layer` uses `DataConfigurator(ConfLayout::PLN)` (plain or linear) data for both input and output.

    Before:

    ```cpp
    ...
    // addConfig({DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});

    ```

    After:

    ```cpp
    addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
    ```

6. The construct is now complete, catching and reporting certain exceptions that may have been thrown before exiting.

    ```cpp
      } catch (InferenceEngine::details::InferenceEngineException &ex) {
        errorMsg = ex.what();
      }
    }
    ```

7. The `execute` method is overridden to implement the functionality of the `cosh` custom layer.  The `inputs` and `outputs` are the data buffers passed as [`Blob`](https://docs.openvinotoolkit.org/2019_R3.1/_docs_IE_DG_Memory_primitives.html) objects.  The template file will simply return `NOT_IMPLEMENTED` by default.  To calculate the `cosh` custom layer, we will replace the `execute` method with the code needed to calculate the `cosh` function in parallel using the [`parallel_for3d`](https://docs.openvinotoolkit.org/2019_R3.1/ie__parallel_8hpp.html) function.

    Before:

    ```cpp
      StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
        ResponseDesc *resp) noexcept override {
        // Add here implementation for layer inference
        // Examples of implementations you can find in Inference Engine tool samples/extensions folder
        return NOT_IMPLEMENTED;
    ```

    After:
    ```cpp
      StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
        ResponseDesc *resp) noexcept override {
        // Add implementation for layer inference here
        // Examples of implementations are in OpenVINO samples/extensions folder

        // Get pointers to source and destination buffers
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        // Get the dimensions from the input (output dimensions are the same)
        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

        // Perform (in parallel) the hyperbolic cosine given by: 
        //    cosh(x) = (e^x + e^-x)/2
        parallel_for3d(N, C, H, [&](int b, int c, int h) {
        // Fill output_sequences with -1
        for (size_t ii = 0; ii < b*c; ii++) {
          dst_data[ii] = (exp(src_data[ii]) + exp(-src_data[ii]))/2;
        }
      });
    return OK;
    }
    ```

#### Edit `CMakeLists.txt`

Because the implementation of the `cosh` custom layer makes use of the parallel processing 
supported by the Inference Engine, we need to add the Intel® Threading Building Blocks 
dependency to `CMakeLists.txt` before compiling.  We will add paths to the header 
and library files and add the Intel® Threading Building Blocks library to the list of link libraries. 
We will also rename the `.so`.

1. Using the text editor, open the CPU extension CMake file `$CLWS/cl_cosh/user_ie_extensions/cpu/CMakeLists.txt`.
2. At the top, rename the `TARGET_NAME` so that the compiled library is named `libcosh_cpu_extension.so`:

    Before:

    ```cmake
    set(TARGET_NAME "user_cpu_extension")
    ```

    After:
    
    ```cmake
    set(TARGET_NAME "cosh_cpu_extension")
    ```

3. We modify the `include_directories` to add the header include path for the Intel® Threading Building Blocks library located in `/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include`:

    Before:

    ```cmake
    include_directories (PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${InferenceEngine_INCLUDE_DIRS}
    )
    ```

    After:
    ```cmake
    include_directories (PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${InferenceEngine_INCLUDE_DIRS}
    "/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include"
    )
    ```

4. We add the `link_directories` with the path to the Intel® Threading Building Blocks library binaries at `/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib`:

    Before:

    ```cmake
    ...
    #enable_omp()
    ```

    After:
    ```cmake
    ...
    link_directories(
    "/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib"
    )
    #enable_omp()
    ```

5. Finally, we add the Intel® Threading Building Blocks library `tbb` to the list of link libraries in `target_link_libraries`:

    Before:

    ```cmake
    target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib})
    ```

    After:

    ```cmake
    target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib} tbb)
    ```

### Compile the Extension Library

To run the custom layer on the CPU during inference, the edited extension C++ source code 
must be compiled to create a `.so` shared library used by the Inference Engine. 
In the following steps, we will now compile the extension C++ library.

1. First, we run the following commands to use CMake to setup for compiling:

    ```bash
    cd $CLWS/cl_cosh/user_ie_extensions/cpu
    mkdir -p build
    cd build
    cmake ..
    ```

    The output will appear similar to:     

    ```
    -- Generating done
    -- Build files have been written to: /home/<user>/cl_tutorial/cl_cosh/user_ie_extensions/cpu/build
    ```

2. The CPU extension library is now ready to be compiled.  Compile the library using the command:

    ```bash
    make -j $(nproc)
    ```

    The output will appear similar to: 

    ```
    [100%] Linking CXX shared library libcosh_cpu_extension.so
    [100%] Built target cosh_cpu_extension
    ```

Move to the next page to continue.


## Execute the Model with the Custom Layer

### Using a C++ Sample

To start on a C++ sample, we first need to build the C++ samples for use with the Inference
Engine:

```bash
cd /opt/intel/openvino/deployment_tools/inference_engine/samples/
./build_samples.sh
```

This will take a few minutes to compile all of the samples.

Next, we will try running the C++ sample without including the `cosh` extension library to see 
the error describing the unsupported `cosh` operation using the command:  

```bash
~/inference_engine_samples_build/intel64/Release/classification_sample_async -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -d CPU
```

The error output will be similar to:

```
[ ERROR ] Unsupported primitive of type: cosh name: ModCosh/cosh/Cosh
```

We will now run the command again, this time with the `cosh` extension library specified 
using the `-l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so` option 
in the command:

```bash
~/inference_engine_samples_build/intel64/Release/classification_sample_async -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -d CPU -l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so
```

The output will appear similar to:

```
Image /home/<user>/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp

classid probability
------- -----------
0       0.9308984  
1       0.0691015

total inference time: xx.xxxxxxx
Average running time of one iteration: xx.xxxxxxx ms

Throughput: xx.xxxxxxx FPS

[ INFO ] Execution successful
```

### Using a Python Sample

First, we will try running the Python sample without including the `cosh` extension library 
to see the error describing the unsupported `cosh` operation using the command:  

```bash
python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -d CPU
```

The error output will be similar to:

```
[ INFO ] Loading network files:
/home/<user>/cl_tutorial/tf_model/model.ckpt.xml
/home/<user>/cl_tutorial/tf_model/model.ckpt.bin
[ ERROR ] Following layers are not supported by the plugin for specified device CPU:
ModCosh/cosh/Cosh, ModCosh/cosh_1/Cosh, ModCosh/cosh_2/Cosh
[ ERROR ] Please try to specify cpu extensions library path in sample's command line parameters using -l or --cpu_extension command line argument
```

We will now run the command again, this time with the `cosh` extension library specified 
using the `-l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so` option 
in the command:

```bash
python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i $CLT/pics/dog.bmp -m $CLWS/cl_ext_cosh/model.ckpt.xml -l $CLWS/cl_cosh/user_ie_extensions/cpu/build/libcosh_cpu_extension.so -d CPU
```

The output will appear similar to:

```
Image /home/<user>/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp

classid probability
------- -----------
0      0.9308984
1      0.0691015
```

**Congratulations!** You have now implemented a custom layer with the Intel® Distribution of OpenVINO™ Toolkit.



## Lesson Glossary: 


Model Optimizer
A command-line tool used for converting a model from one of the supported frameworks to an Intermediate Representation (IR), including certain performance optimizations, that is compatible with the Inference Engine.

Optimization Techniques
Optimization techniques adjust the original trained model in order to either reduce the size of or increase the speed of a model in performing inference. Techniques discussed in the lesson include quantization, freezing and fusion.

Quantization
Reduces precision of weights and biases (to lower precision floating point values or integers), thereby reducing compute time and size with some (often minimal) loss of accuracy.

Freezing
In TensorFlow this removes metadata only needed for training, as well as converting variables to constants. Also a term in training neural networks, where it often refers to freezing layers themselves in order to fine tune only a subset of layers.

Fusion
The process of combining certain operations together into one operation and thereby needing less computational overhead. For example, a batch normalization layer, activation layer, and convolutional layer could be combined into a single operation. This can be particularly useful for GPU inference, where the separate operations may occur on separate GPU kernels, while a fused operation occurs on one kernel, thereby incurring less overhead in switching from one kernel to the next.

Supported Frameworks
The Intel® Distribution of OpenVINO™ Toolkit currently supports models from five frameworks (which themselves may support additional model frameworks): Caffe, TensorFlow, MXNet, ONNX, and Kaldi.

Caffe
The “Convolutional Architecture for Fast Feature Embedding” (CAFFE) framework is an open-source deep learning library originally built at UC Berkeley.

TensorFlow
TensorFlow is an open-source deep learning library originally built at Google. As an Easter egg for anyone who has read this far into the glossary, this was also your instructor’s first deep learning framework they learned, back in 2016 (pre-V1!).

MXNet
Apache MXNet is an open-source deep learning library built by Apache Software Foundation.

ONNX
The “Open Neural Network Exchange” (ONNX) framework is an open-source deep learning library originally built by Facebook and Microsoft. PyTorch and Apple-ML models are able to be converted to ONNX models.

Kaldi
While still open-source like the other supported frameworks, Kaldi is mostly focused around speech recognition data, with the others being more generalized frameworks.

Intermediate Representation
A set of files converted from one of the supported frameworks, or available as one of the Pre-Trained Models. This has been optimized for inference through the Inference Engine, and may be at one of several different precision levels. Made of two files:

.xml - Describes the network topology
.bin - Contains the weights and biases in a binary file
Supported Layers
Layers supported for direct conversion from supported framework layers to intermediate representation layers through the Model Optimizer. While nearly every layer you will ever use is in the supported frameworks is supported, there is sometimes a need for handling Custom Layers.

Custom Layers
Custom layers are those outside of the list of known, supported layers, and are typically a rare exception. Handling custom layers in a neural network for use with the Model Optimizer depends somewhat on the framework used; other than adding the custom layer as an extension, you otherwise have to follow [instructions](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html) specific to the framework.




