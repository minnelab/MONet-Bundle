FROM nvcr.io/nvidia/pytorch:25.02-py3

#RUN ltt install torch==2.4.0
RUN pip install git+https://github.com/SimoneBendazzoli93/dynamic-network-architectures.git
RUN pip install git+https://github.com/SimoneBendazzoli93/nnUNet.git
RUN pip install git+https://github.com/SimoneBendazzoli93/MONAI.git@dev
RUN pip install monet-bundle


WORKDIR /workspace

COPY run.sh .

ENTRYPOINT ["/bin/bash", "run.sh"]
