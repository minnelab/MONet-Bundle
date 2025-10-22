# MAIA Segmentation Portal

**MAIA Segmentation Portal** is a user-friendly portal, hosted in [MAIA](https://maia.app.cloud.cbh.kth.se/maia), that allows users to interact with the available models, upload medical images, and receive predictions in seconds. Alternatively, users can download the models and run inference on their own machines with a single command.
The models are available in the MONet Bundle format, which is compatible with MONAI Deploy and MONAI Label.
The single models are deployed and made available as single MONAI Label server applications, hosted in **MAIA**.
The **MAIA Segmentation Portal** is running as a KubeFlow-based project, so it is possible to run inference on the models from the KubeFlow UI.
If you want to create your own MONet-based model, we provide instructions on how to run the training, starting from your annotated data, either locally or with KubeFlow, using the MONet Bundle format.

## GUI Installation
To start using the MAIA Segmentation Portal, you can install the GUI application.
The GUI application is available for Windows and Linux.

Download the Windows version of the MAIA Segmentation Portal from [here](https://github.com/minnelab/MONet-Bundle/releases/download/v1.3/MAIA_Segmentation_Portal.exe)
or the Linux version from [here](https://github.com/minnelab/MONet-Bundle/releases/download/v1.3/MAIA_Segmentation_Portal_Linux)

## Installation
To install the MAIA Segmentation Portal Python API:
```bash
pip install monet-bundle
```

## Get Access to the Portal
To access the MAIA Segmentation Portal, you need to register for an account in MAIA and request access to the *maia-segmentation* project in the [MAIA - Sign Up Page](https://maia.app.cloud.cbh.kth.se/maia/register/).
Once your registration is approved, you will be able to log in to the portal and start using the models.
To log in to the portal, you can use the following command:
```bash
MONet_login --username <USERNAME>
```
You will be prompted to enter your password, and then you will be able to access the portal.
If your access token is expired, you will be prompted to enter your username and password again.

## List Available Models
To list the available models in the MAIA Segmentation Portal, you can use the following command:
```bash
MONet_login --username <USERNAME> --list-models
```

## Run Remote Inference
To run remote inference on the MAIA Segmentation Portal, you can use the `MONet_remote_inference` command.
This command allows you to upload your medical images to the portal and receive predictions in seconds.
```bash
MONet_remote_inference --model <MODEL_NAME> -i <INPUT_FILE> -o <OUTPUT_FILE>
```

## Run Local Inference
To run local inference with the MONet Bundle, you can use the `MONet_local_inference` command.
This command allows you to run inference on your own machine with a single command.
The pre-requisites for running local inference are:
- Available GPU with NVIDIA drivers installed
- PyTorch >=2.4.1 installed

Recommended pytorch installation command:
```bash
pip install light-the-torch
ltt install torch==2.4.1 torchvision
```

```bash
MONet_local_inference --model <MODEL_NAME> -i <INPUT_FILE> -o <OUTPUT_FILE>
```