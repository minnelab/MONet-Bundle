# Federated Brain Tumor Segmentation (FedBraTS)

[![Docker Pulls](https://img.shields.io/docker/pulls/maiacloud/brats-x64-workstation-dgpu-linux-amd64)](https://hub.docker.com/r/maiacloud/brats-x64-workstation-dgpu-linux-amd64)

This repository provides the necessary tools and instructions to run federated brain tumor segmentation, described in the paper [MONet-FL: Extending nnU-Net with MONAI for Clinical Federated Learning](). 

The Federated training is performed following the tutorial [MONet-FL](https://github.com/minnelab/MONet-Bundle/blob/main/MONet-FL.ipynb).


Data used in this study were obtained as part of the Challenge project through Synapse ID (syn64153430).

In detail, the 3 BraTS datasets used in the study are:
- [BraTS 2025-GLI Pre-treatment](https://www.synapse.org/Synapse:syn65773245)
- [BraTS 2025-MEN](https://www.synapse.org/Synapse:syn64952505)
- [BraTS 2025-MET](https://www.synapse.org/Synapse:syn64951882)

![](./images/Dataset.png)
Each dataset is assigned to a different site, and the federated training is performed using the [MONet bundle](https://github.com/minnelab/MONet-Bundle).


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

## Results

| **Dataset - Label**      | **Experiment**       | **DSC**         | **ASD \[mm]**  |
| ------------------------ | -------------------- | --------------- | -------------- |
| BraTS-GLI (n=251) - ET   | Cross-Site BraTS-MEN | 0.695 ± 0.292   | 3.05 ± 8.69    |
|                          | Cross-Site BraTS-MET | 0.729 ± 0.271   | 1.62 ± 4.97    |
|                          | *MONet-FL*           | *0.788 ± 0.261* | *1.57 ± 5.56*  |
|                          | BraTSFusion          | 0.774 ± 0.271   | 1.61 ± 4.07    |
|                          | nnU-Net-Baseline     | 0.796 ± 0.254   | 1.38 ± 3.72    |
| BraTS-GLI (n=251) - NETC | Cross-Site BraTS-MEN | 0.709 ± 0.296   | 2.17 ± 7.84    |
|                          | Cross-Site BraTS-MET | 0.787 ± 0.218   | 1.47 ± 2.92    |
|                          | *MONet-FL*           | *0.836 ± 0.178* | *0.88 ± 1.83*  |
|                          | BraTSFusion          | 0.846 ± 0.153   | 0.84 ± 1.40    |
|                          | nnU-Net-Baseline     | 0.871 ± 0.143   | 0.81 ± 1.73    |
| BraTS-GLI (n=251) - SNFH | Cross-Site BraTS-MEN | 0.823 ± 0.197   | 2.52 ± 8.67    |
|                          | Cross-Site BraTS-MET | 0.833 ± 0.167   | 1.29 ± 3.35    |
|                          | *MONet-FL*           | *0.884 ± 0.126* | *0.79 ± 2.21*  |
|                          | BraTSFusion          | 0.897 ± 0.118   | 1.92 ± 0.08    |
|                          | nnU-Net-Baseline     | 0.896 ± 0.116   | 0.48 ± 1.38    |
| BraTS-MEN (n=200) - ET   | Cross-Site BraTS-GLI | 0.368 ± 0.352   | 8.92 ± 7.78    |
|                          | Cross-Site BraTS-MET | 0.366 ± 0.338   | 8.85 ± 12.57   |
|                          | *MONet-FL*           | *0.422 ± 0.354* | *7.14 ± 7.38*  |
|                          | BraTSFusion          | 0.304 ± 0.308   | 8.45 ± 8.67    |
|                          | nnU-Net-Baseline     | 0.367 ± 0.337   | 9.52 ± 8.49    |
| BraTS-MEN (n=200) - NETC | Cross-Site BraTS-GLI | 0.716 ± 0.305   | 6.41 ± 12.83   |
|                          | Cross-Site BraTS-MET | 0.714 ± 0.299   | 4.58 ± 7.08    |
|                          | *MONet-FL*           | *0.774 ± 0.273* | *3.25 ± 5.95*  |
|                          | BraTSFusion          | 0.728 ± 0.349   | 3.97 ± 8.76    |
|                          | nnU-Net-Baseline     | 0.774 ± 0.312   | 1.87 ± 3.28    |
| BraTS-MEN (n=200) - SNFH | Cross-Site BraTS-GLI | 0.763 ± 0.363   | 2.18 ± 10.75   |
|                          | Cross-Site BraTS-MET | 0.745 ± 0.345   | 3.47 ± 13.43   |
|                          | *MONet-FL*           | *0.911 ± 0.184* | *2.29 ± 12.21* |
|                          | BraTSFusion          | 0.925 ± 0.140   | 2.05 ± 10.11   |
|                          | nnU-Net-Baseline     | 0.924 ± 0.168   | 2.35 ± 10.16   |
| BraTS-MET (n=48) - ET    | Cross-Site BraTS-GLI | 0.560 ± 0.335   | 5.97 ± 12.76   |
|                          | Cross-Site BraTS-MEN | 0.358 ± 0.362   | 11.55 ± 23.56  |
|                          | *MONet-FL*           | *0.634 ± 0.303* | *3.47 ± 8.86*  |
|                          | BraTSFusion          | 0.506 ± 0.313   | 6.79 ± 18.47   |
|                          | nnU-Net-Baseline     | 0.677 ± 0.281   | 3.14 ± 8.02    |
| BraTS-MET (n=48) - NETC  | Cross-Site BraTS-GLI | 0.658 ± 0.314   | 3.68 ± 10.01   |
|                          | Cross-Site BraTS-MEN | 0.547 ± 0.376   | 8.59 ± 19.90   |
|                          | *MONet-FL*           | *0.719 ± 0.262* | *1.48 ± 1.92*  |
|                          | BraTSFusion          | 0.698 ± 0.256   | 1.42 ± 1.27    |
|                          | nnU-Net-Baseline     | 0.727 ± 0.251   | 2.19 ± 5.06    |
| BraTS-MET (n=48) - SNFH  | Cross-Site BraTS-GLI | 0.613 ± 0.299   | 0.62 ± 0.36    |
|                          | Cross-Site BraTS-MEN | 0.541 ± 0.338   | 13.51 ± 25.83  |
|                          | *MONet-FL*           | *0.763 ± 0.156* | *0.60 ± 0.66*  |
|                          | BraTSFusion          | 0.717 ± 0.194   | 2.78 ± 11.49   |
|                          | nnU-Net-Baseline     | 0.791 ± 0.147   | 0.53 ± 0.44    |

### References

    [1] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.
    [2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
    [3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
    [4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q
    [5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF
    [6] LaBella, D., Adewole, M., Alonso-Basanta, M., Altes, T., Anwar, S. M., Baid, U., Bergquist, T., Bhalerao, R., Chen, S., Chung, V., Conte, G.-M., Dako, F., Eddy, J., Ezhov, I., Godfrey, D., Hilal, F., Familiar, A., Farahani, K., Iglesias, J. E., … Calabrese, E. (2023). The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023: Intracranial Meningioma (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2305.07642
    [7] Moawad, A. W., Janas, A., Baid, U., Ramakrishnan, D., Saluja, R., Ashraf, N., Maleki, N., Jekel, L., Yordanov, N., Fehringer, P., Gkampenis, A., Amiruddin, R., Manteghinejad, A., Adewole, M., Albrecht, J., Anazodo, U., Aneja, S., Anwar, S. M., Bergquist, T., … Aboian, M. (2023). The Brain Tumor Segmentation (BraTS-METS) Challenge 2023: Brain Metastasis Segmentation on Pre-treatment MRI (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2306.00838

