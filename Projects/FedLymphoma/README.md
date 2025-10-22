# Federated Lymphoma Segmentation (FedLymphoma)

[![Docker Pulls](https://img.shields.io/docker/pulls/maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64)](https://hub.docker.com/r/maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64)

This repository provides the necessary tools and instructions to run federated lymphoma segmentation, described in the paper [MONet-FL: Extending nnU-Net with MONAI for Clinical Federated Learning](). 

The Federated training is performed following the tutorial [MONet-FL](https://github.com/minnelab/MONet-Bundle/blob/main/MONet-FL.ipynb).


In detail, the 2 Lymphoma datasets used in the study are:
- [AutoPET](https://zenodo.org/records/10990932)
- IndolentLymphoma-KUH, private collection of PET-CT of lymphoma patients and corresponding annotations from Karolinska University Hospital, Stockholm, Sweden

![](./images/Dataset.png)
Each dataset is assigned to a different site, and the federated training is performed using the [MONet bundle](https://github.com/minnelab/MONet-Bundle).

## FedLymphoma Inference on PET-CT [DICOM Version]

The input folder should contain all the DICOM files of the study you want to process (with the CT and PT series), and the output folder will contain the lymphoma lesion predictions in DICOM SEG format, and an additional STL file with the 3D mesh of the lymphoma lesion segmentation.


```bash
docker run --rm -it --gpus=all --shm-size 2G -v $INPUT_STUDY_FOLDER:/var/holoscan/input -v $PREDICTIONS_FOLDER:/var/holoscan/output maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64:1.0
```

## FedLymphoma Inference on PET-CT [NIFTI Version]

The input folder should contain all the NIfTI files of the study you want to process (one per modality, with the given suffix identifier), and the output folder will contain the predictions in NIfTI format.
The suffix identifiers for the modalities are:
- `_ct.nii.gz` for the CT modality
- `_pet.nii.gz` for the PET modality


```bash
docker run --rm -it --gpus=all --shm-size 2G -v $INPUT_STUDY_FOLDER:/var/holoscan/input -v $PREDICTIONS_FOLDER:/var/holoscan/output maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64:1.0-nifti
```

## FedLymphoma Inference on PET-CT [XNAT Integration]

To run the inference on a study stored in XNAT, using the [XNAT Container Service Plugin](https://wiki.xnat.org/container-service) you can use the following command:
```json
{
    "name": "lymphoma-segmentation",
    "label": "lymphoma-segmentation",
    "description": "Runs Lymphoma Segmentation on a session derived PET-CT scan.",
    "version": "1.0",
    "schema-version": "1.0",
    "info-url": "https://github.com/rordenlab/dcm2niix",
    "image": "maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64:1.0-xnat",
    "type": "docker",
    "command-line": "",
    "mounts": [
      {
        "name": "dicom-in-ct",
        "writable": true,
        "path": "/var/holoscan/input/ct"
      },
      {
        "name": "dicom-in-pt",
        "writable": true,
        "path": "/var/holoscan/input/pt"
      },
      {
        "name": "dicom-out",
        "writable": true,
        "path": "/var/holoscan/output"
      }
    ],
    "environment-variables": {
      "ORTHANC_URL": "orthancURL"
    },
    "ports": {},
    "inputs": [
      {
        "name": "orthanc-url",
        "description": "Orthanc URL",
        "type": "string",
        "required": true,
        "replacement-key": "orthancURL",
        "select-values": []
      }
    ],
    "outputs": [
      {
        "name": "dicom-seg",
        "description": "The output segmentation",
        "required": true,
        "mount": "dicom-out"
      }
    ],
    "xnat": [
      {
        "name": "lymphoma-segmentation",
        "label": "lymphoma-segmentation",
        "description": "Runs Lymphoma Segmentation on a session derived scan.",
        "contexts": [
          "xnat:imageSessionData"
        ],
        "external-inputs": [
          {
            "name": "session",
            "description": "Input session",
            "type": "Session",
            "matcher": "'DICOM' in @.scans[*].resources[*].label",
            "required": true,
            "load-children": true
          }
        ],
        "derived-inputs": [
          {
            "name": "ct-scan",
            "description": "Input CT scan",
            "type": "Scan",
            "matcher": "@.modality == 'CT'",
            "required": true,
            "load-children": true,
            "derived-from-wrapper-input": "session",
            "multiple": false
          },
          {
            "name": "ct-scan-dicoms",
            "description": "The CT dicom resource on the scan",
            "type": "Resource",
            "matcher": "@.label == 'DICOM'",
            "required": true,
            "provides-files-for-command-mount": "dicom-in-ct",
            "load-children": true,
            "derived-from-wrapper-input": "ct-scan",
            "multiple": false
          },
          {
            "name": "pt-scan",
            "description": "Input PET scan",
            "type": "Scan",
            "matcher": "@.modality == 'PT'",
            "required": true,
            "load-children": true,
            "derived-from-wrapper-input": "session",
            "multiple": false
          },
          {
            "name": "pt-scan-dicoms",
            "description": "The PET dicom resource on the scan",
            "type": "Resource",
            "matcher": "@.label == 'DICOM'",
            "required": true,
            "provides-files-for-command-mount": "dicom-in-pt",
            "load-children": true,
            "derived-from-wrapper-input": "pt-scan",
            "multiple": false
          }
        ],
        "output-handlers": [
          {
            "name": "dicom-seg-resource",
            "accepts-command-output": "dicom-seg",
            "as-a-child-of": "ct-scan",
            "type": "Resource",
            "label": "DICOM-SEG",
            "format": "DICOM",
            "tags": []
          }
        ]
      }
    ],
    "shm-size": 1073741824,
    "container-labels": {},
    "generic-resources": {
      "nvidia.com/gpu": "1"
    },
    "ulimits": {},
    "secrets": []
  }
  ```
  The additional environment variable `ORTHANC_URL` should be set to the URL of the Orthanc server where to send the DICOM SEG files produced by the inference.


## Results

| **Dataset - Label**        | **Experiment**       | **DSC**            | **ASD [mm]**         |
|---------------------------|----------------------|--------------------|----------------------|
| **AutoPET (n=27) - Lesion** | Cross-Site IL1        | 0.445 ± 0.270       | 50.19 ± 95.19        |
|                           | Cross-Site IL2        | 0.587 ± 0.251       | 40.06 ± 89.72        |
|                           | *MONet-FL*            | *0.694 ± 0.279*     | *10.80 ± 24.61*      |
|                           | LymphoFusion          | 0.756 ± 0.244       | 19.27 ± 39.78        |
|                           | nnU-Net-Baseline      | 0.758 ± 0.252       | 38.50 ± 133.01       |
| **IL1 (n=16) - Lesion**     | Cross-Site AutoPET    | 0.317 ± 0.255       | 99.46 ± 147.08       |
|                           | Cross-Site IL2        | 0.297 ± 0.230       | 111.11 ± 172.99      |
|                           | *MONet-FL*            | *0.453 ± 0.252*     | *70.91 ± 127.18*     |
|                           | LymphoFusion          | 0.421 ± 0.223       | 90.29 ± 143.43       |
|                           | nnU-Net-Baseline      | 0.512 ± 0.215       | 12.95 ± 11.69        |
| **IL2 (n=31) - Lesion**     | Cross-Site AutoPET    | 0.534 ± 0.265       | 39.34 ± 71.36        |
|                           | Cross-Site IL1        | 0.391 ± 0.247       | 40.03 ± 79.63        |
|                           | *MONet-FL*            | *0.610 ± 0.234*     | *25.03 ± 64.44*      |
|                           | LymphoFusion          | 0.665 ± 0.242       | 23.07 ± 70.48        |
|                           | nnU-Net-Baseline      | 0.671 ± 0.275       | 24.09 ± 72.41        |

**Table 1: Dice Similarity Coefficient (DSC) and Average Surface Distance (ASD) for each Lymphoma experiment across the three datasets (over the validation set): AutoPET, Indolent Lymphoma 1 (IL1) and Indolent Lymphoma 2 (IL2).**


### References

    [1] Ingrisch, M., Dexl, J., Jeblick, K., Cyran, C., Gatidis, S., & Kuestner, T. (2024). Automated Lesion Segmentation in Whole-Body PET/CT - Multitracer Multicenter generalization. 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024). Zenodo. https://doi.org/10.5281/zenodo.10990932
    [2] Gatidis S, Kuestner T. (2022) A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions) [Dataset]. The Cancer Imaging Archive. DOI: 10.7937/gkr0-xv29 