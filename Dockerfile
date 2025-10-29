FROM nvcr.io/nvidia/pytorch:25.02-py3

#RUN ltt install torch==2.4.0
RUN pip install git+https://github.com/SimoneBendazzoli93/dynamic-network-architectures.git
RUN pip install git+https://github.com/SimoneBendazzoli93/nnUNet.git
RUN pip install git+https://github.com/SimoneBendazzoli93/MONAI.git@dev
RUN pip install monet-bundle

ARG MONAILABEL=false
RUN if [ "$MONAILABEL" = "true" ]; then \
    pip install pytorch-ignite SimpleITK nibabel nilearn git+https://github.com/minnelab/MONet-Bundle.git git+https://github.com/SimoneBendazzoli93/MONAILabel.git pydicom==2.4.4; \
    fi
WORKDIR /workspace

COPY run.sh .

ENTRYPOINT ["/bin/bash", "run.sh"]
