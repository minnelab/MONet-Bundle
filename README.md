# MONet Bundle

[![Build](https://github.com/SimoneBendazzoli93/MONet-Bundle/actions/workflows/build.yaml/badge.svg)](https://github.com/SimoneBendazzoli93/MONet-Bundle/actions/workflows/build.yaml)

[![Documentation Status](https://readthedocs.org/projects/monet-bundle/badge/?version=latest)](https://monet-bundle.readthedocs.io/en/latest/?badge=latest)
![Version](https://img.shields.io/badge/MONet-v1.0-blue)
[![License](https://img.shields.io/badge/license-GPL%203.0-green.svg)](https://opensource.org/licenses/GPL-3.0)
![Python](https://img.shields.io/badge/python-3.10+-orange)


![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/SimoneBendazzoli93/MONet-Bundle?logo=github)
![GitHub contributors](https://img.shields.io/github/contributors/SimoneBendazzoli93/MONet-Bundle?logo=github)
![GitHub top language](https://img.shields.io/github/languages/top/SimoneBendazzoli93/MONet-Bundle?logo=github)
![GitHub language count](https://img.shields.io/github/languages/count/SimoneBendazzoli93/MONet-Bundle?logo=github)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/SimoneBendazzoli93/MONet-Bundle/publish_release.yaml?logo=github)
![GitHub all releases](https://img.shields.io/github/downloads/SimoneBendazzoli93/MONet-Bundle/total?logo=github)
![PyPI - Downloads](https://img.shields.io/pypi/dm/monet-bundle?logo=pypi)
![GitHub](https://img.shields.io/github/license/SimoneBendazzoli93/MONet-Bundle?logo=github)
![PyPI - License](https://img.shields.io/pypi/l/monet-bundle?logo=pypi)


![GitHub repo size](https://img.shields.io/github/repo-size/SimoneBendazzoli93/MONet-Bundle?logo=github)
![GitHub release (with filter)](https://img.shields.io/github/v/release/SimoneBendazzoli93/MONet-Bundle?logo=github)
![PyPI](https://img.shields.io/pypi/v/monet-bundle?logo=pypi)

This repository contains the implementation of the MONet Bundle, with some instructions on how to use it and how to convert a generic nnUNet model to MONAI Bundle format.

For more details about the MONet Bundle, please refer to the Jupyter notebook [MONet_Bundle.ipynb](./MONet_Bundle.ipynb).


## 2025-06-25 UPDATE: Check out the MONet Bundle for FedBraTS and FedLymphoma!
The MONet Bundle has been used in the Federated Brain Tumor Segmentation [FedBraTS](./Projects/FedBraTS/README.md) and Federated Lymphoma Segmentation [FedLymphoma](./Projects/FedLymphoma/README.md) projects, which are described in the following paper:
- [MONet-FL: Extending nnU-Net with MONAI for Clinical Federated Learning]()


## Download the MONet Bundle
You can download the MONet Bundle from the following link: [MONet Bundle](https://raw.githubusercontent.com/SimoneBendazzoli93/MONet-Bundle/main/MONetBundle.zip)
ALternatively, you can use the following command to download the MONet Bundle:
```bash
wget https://raw.githubusercontent.com/SimoneBendazzoli93/MONet-Bundle/main/MONetBundle.zip
```
or, through the Python Script:
```bash
MONet_fetch_bundle.py --bundle_path <FOLDER_PATH>
``` 

## Convert a trained nnUNet model to MONAI Bundle

To convert a trained nnUNet model to MONAI Bundle format, you can start with exporting a nnUNet trained model  with the `nnUNetv2_export_model_to_zip` command. This command will export the model to a zip file that can be used in the conversion process.
```bash
nnUNetv2_export_model_to_zip -d 009 -o Task09_Spleen.zip -c 3d_fullres -tr nnUNetTrainer -p nnUNetPlans -chk checkpoint_final.pth checkpoint_best.pth --not_strict
```
For testing purposes, you can use the `Task09_Spleen.zip` file provided in this repository: https://github.com/SimoneBendazzoli93/nnUNet-MONAI-Bundle/releases/download/v1.0/Task09_Spleen.zip. This file contains a trained nnUNet model for the Spleen segmentation task, for only the `3d_fullres` configuration and the fold `0`.



Next, you can build the provided Docker image to convert the model to MONAI Bundle format. The Dockerfile is provided in this repository, and you can build the image with the following command:

```bash
docker build -t nnunet-monai-bundle-converter .
```
The converter will first convert the nnUNet model to MONAI Bundle format, and then create the corresponding TorchScript model, which can be used for inference with MONAI Deploy.
For testing purposes, you can use the `Task09_Spleen.zip` file provided in this repository.

To run the conversion, you can use the following command:
```bash
wget https://github.com/SimoneBendazzoli93/MONet-Bundle/releases/download/v1.0/Task09_Spleen.zip
python MONet_run_conversion.py --bundle_path <MONAI_BUNDLE_PATH> --nnunet_model <NNUNET_CHECKPOINT_PATH>.zip
```

## Package the MONet Bundle with MONAI Deploy
To package the MONet Bundle with MONAI Deploy, you can use the `monai-deploy package` command. This command will create a deployable bundle that can be used for inference with MONAI Deploy.

```bash
monai-deploy package examples/apps/spleen_nnunet_seg_app -c examples/apps/spleen_nnunet_seg_app.yaml -t spleen:1.0 --platform x86_64
```

## Run inference with MONAI Deploy

The resulting Docker context can be found in the `deploy/spleen-x64-workstation-dgpu-linux-amd64:1.0` directory. You can use this context to build a Docker image that can be used for inference with MONAI Deploy:
```bash
# Copy the TorchScript model to the Docker context
cp nnUNetBundle/models/fold_0/model.ts deploy/spleen-x64-workstation-dgpu-linux-amd64:1.0/models/model/

docker build deploy/spleen-x64-workstation-dgpu-linux-amd64:1.0 --build-arg UID=1000 --build-arg GID=1000 --build-arg UNAME=holoscan -f deploy/spleen-x64-workstation-dgpu-linux-amd64:1.0/Dockerfile -t spleen-x64-workstation-dgpu-linux-amd64:1.0
```

To test the resulting Docker image, you can run:
```bash
MONet_inference_dicom.py --dicom_study_folder <INPUT_FOLDER> --prediction_output_folder <OUTPUT_DIR> --docker-image maiacloud/spleen-x64-workstation-dgpu-linux-amd64:1.0
```
Specifying the input and output folders, together with the TorchScript model path.
The input folder should contain all the DICOM files of the study you want to process, and the output folder will contain the predictions in DICOM SEG format, and an additional STL file with the 3D mesh of the segmentation.



To create the same Docker image running inference on NIFTI images, you can use the provided `Dockerfile` in the `deploy/spleen-x64-workstation-dgpu-linux-amd64:1.0-nifti` directory. The Dockerfile is already set up to run inference on NIfTI images, and it includes the necessary dependencies.
To test the resulting Docker image, you can run:
```bash
MONet_inference_nifti.py --study_folder <INPUT_FOLDER> --prediction_output_folder <OUTPUT_DIR> --docker-image maiacloud/spleen-x64-workstation-dgpu-linux-amd64:1.0-nifti
```
Specifying the input and output folders, together with the TorchScript model path.
The input folder should contain all the NIfTI files of the study you want to process (one per modality, with the given suffix identifier), and the output folder will contain the predictions in NIfTI format.
