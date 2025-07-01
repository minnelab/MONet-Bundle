FROM maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64:1.1

USER root
RUN mkdir /opt/holoscan/monailabel
RUN chmod -R 777 /opt/holoscan/monailabel

RUN apt update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
    wget \
    unzip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

USER $UNAME

ENV PATH=/home/${UNAME}/.local/bin:/opt/nvidia/holoscan/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/${UNAME}/.local/lib/python3.10/site-packages/holoscan/lib

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

RUN pip install git+https://github.com/SimoneBendazzoli93/MONAILabel.git@eff0740cda2bda5cd424d4e9e69574fbe7edca9e

RUN cd /opt/holoscan/monailabel && monailabel apps --download --name monaibundle --output .

RUN pip install pydicom==2.4.4 nilearn

RUN mkdir -p /opt/holoscan/monailabel/monaibundle/model
RUN cd /opt/holoscan/monailabel/monaibundle/model && wget https://raw.githubusercontent.com/SimoneBendazzoli93/MONet-Bundle/main/MONetBundle.zip
RUN unzip /opt/holoscan/monailabel/monaibundle/model/MONetBundle.zip -d /opt/holoscan/monailabel/monaibundle/model/ 

ENTRYPOINT ["/var/holoscan/tools"]