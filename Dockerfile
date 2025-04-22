FROM nvcr.io/nvidia/pytorch:24.05-py3

#RUN ltt install torch==2.4.0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace
COPY convert_nnunet_checkpoint_to_bundle_checkpoint.py .
RUN chmod +x convert_nnunet_checkpoint_to_bundle_checkpoint.py

COPY convert_ckpt_to_ts.py .
RUN chmod +x convert_ckpt_to_ts.py
COPY run.sh .

ENTRYPOINT ["/bin/bash", "run.sh"]
