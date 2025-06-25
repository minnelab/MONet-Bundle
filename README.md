# MONet Bundle

This repository contains the implementation of the MONet Bundle, with some instructions on how to use it and how to convert a generic nnUNet model to MONAI Bundle format.

For more details about the MONet Bundle, please refer to the Jupyter notebook [MONet_Bundle.ipynb](MONet_Bundle.ipynb).


## 2025-06-25 UPDATE: Check out the MONet Bundle for FedBraTS and FedLymphoma!
The MONet Bundle has been used in the Federated Brain Tumor Segmentation [FedBraTS](./Projects/FedBraTS/) and Federated Lymphoma Segmentation [FedLymphoma](./Projects/FedLymphoma/) projects, which are described in the following paper:
- [MONet-FL: Extending nnU-Net with MONAI for Clinical Federated Learning]()


## Download the MONet Bundle
You can download the MONet Bundle from the following link: [MONet Bundle](https://raw.githubusercontent.com/SimoneBendazzoli93/MONet-Bundle/main/MONetBundle.zip)

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
For testing purposes, you can use the `Task09_Spleen.zip` file provided in this repository and the [MONet Bundle template](./MONetBundle/).

The instructions to run the converter can be found in the [run_conversion.py](run_conversion.py) file.

## Package the MONet Bundle with MONAI Deploy
To package the MONet Bundle with MONAI Deploy, you can use the `monai-deploy package` command. This command will create a deployable bundle that can be used for inference with MONAI Deploy.

```bash
monai-deploy package examples/apps/spleen_nnunet_seg_app -c examples/apps/spleen_nnunet_seg_appapp.yaml -t spleen:1.0 --platform x86_64
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
python run_inference_dicom.py
```
Specifying the input and output folders, together with the TorchScript model path.
The input folder should contain all the DICOM files of the study you want to process, and the output folder will contain the predictions in DICOM SEG format, and an additional STL file with the 3D mesh of the segmentation.



To create the same Docker image running inference on NIFTI images, you can use the provided `Dockerfile` in the `deploy/spleen-x64-workstation-dgpu-linux-amd64:1.0-nifti` directory. The Dockerfile is already set up to run inference on NIfTI images, and it includes the necessary dependencies.
To test the resulting Docker image, you can run:
```bash
python run_inference_nifti.py
```
Specifying the input and output folders, together with the TorchScript model path.
The input folder should contain all the NIfTI files of the study you want to process (one per modality, with the given suffix identifier), and the output folder will contain the predictions in NIfTI format.
