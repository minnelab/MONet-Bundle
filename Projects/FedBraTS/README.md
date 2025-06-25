# Federated Brain Tumor Segmentation (FedBraTS)

[![Docker Pulls](https://img.shields.io/docker/pulls/maiacloud/brats-x64-workstation-dgpu-linux-amd64)](https://hub.docker.com/r/maiacloud/brats-x64-workstation-dgpu-linux-amd64)

This repository provides the necessary tools and instructions to run federated brain tumor segmentation, described in the paper [MONet-FL: Extending nnU-Net with MONAI for Clinical Federated Learning](). 

The Federated training is performed following the tutorial [MONet-FL](https://github.com/SimoneBendazzoli93/MONet-Bundle/blob/main/MONet-FL.ipynb).


Data used in this study were obtained as part of the Challenge project through Synapse ID (syn64153430).

In detail, the 3 BraTS datasets used in the study are:
- [BraTS 2025-GLI Pre-treatment](https://www.synapse.org/Synapse:syn65773245)
- [BraTS 2025-MEN](https://www.synapse.org/Synapse:syn64952505)
- [BraTS 2025-MET](https://www.synapse.org/Synapse:syn64951882)

![](./images/Dataset.png)
Each dataset is assigned to a different site, and the federated training is performed using the [MONet bundle](https://github.com/SimoneBendazzoli93/MONet-Bundle).


## FedBraTS Inference on Multimodal MRI [NIFTI Version]

The input folder should contain all the NIfTI files of the study you want to process (one per modality, with the given suffix identifier), and the output folder will contain the predictions in NIfTI format.
The suffix identifiers for the modalities are:
- `-t1c.nii.gz` for the T1 post-contrast modality
- `-t1n.nii.gz` for the T1 modality
- `-t2f.nii.gz` for the T2 Flair modality
- `-t2w.nii.gz` for the T2 modality


```bash
docker run --rm -it --gpus=all --shm-size 2G -v $INPUT_STUDY_FOLDER:/var/holoscan/input -v $PREDICTIONS_FOLDER:/var/holoscan/output maiacloud/brats-x64-workstation-dgpu-linux-amd64:1.0-nifti
```

### References

    [1] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.
    [2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
    [3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
    [4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q
    [5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF
    [6] LaBella, D., Adewole, M., Alonso-Basanta, M., Altes, T., Anwar, S. M., Baid, U., Bergquist, T., Bhalerao, R., Chen, S., Chung, V., Conte, G.-M., Dako, F., Eddy, J., Ezhov, I., Godfrey, D., Hilal, F., Familiar, A., Farahani, K., Iglesias, J. E., … Calabrese, E. (2023). The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023: Intracranial Meningioma (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2305.07642
    [7] Moawad, A. W., Janas, A., Baid, U., Ramakrishnan, D., Saluja, R., Ashraf, N., Maleki, N., Jekel, L., Yordanov, N., Fehringer, P., Gkampenis, A., Amiruddin, R., Manteghinejad, A., Adewole, M., Albrecht, J., Anazodo, U., Aneja, S., Anwar, S. M., Bergquist, T., … Aboian, M. (2023). The Brain Tumor Segmentation (BraTS-METS) Challenge 2023: Brain Metastasis Segmentation on Pre-treatment MRI (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2306.00838

