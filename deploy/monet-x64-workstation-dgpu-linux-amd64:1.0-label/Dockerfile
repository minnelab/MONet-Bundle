FROM maiacloud/spleen-x64-workstation-dgpu-linux-amd64:1.0

USER root
RUN mkdir /opt/holoscan/monailabel
RUN chmod -R 777 /opt/holoscan/monailabel

RUN apt update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
    build-essential \
    python3.10-dev  \
    wget \
    unzip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

USER $UNAME

ENV PATH=/home/${UNAME}/.local/bin:/opt/nvidia/holoscan/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/${UNAME}/.local/lib/python3.10/site-packages/holoscan/lib

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

SHELL ["/bin/bash", "-c"]
# Install NVM
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
    && source $NVM_DIR/nvm.sh \
    && nvm install 18 \
    && nvm alias default 18 \
    && nvm use default

# Add Node.js and npm to PATH for future RUNs and the container environment
ENV NODE_VERSION=18
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH


RUN git clone https://github.com/SimoneBendazzoli93/MONAILabel.git && cd MONAILabel && git checkout b15ca19f97bae212a93bb66c5de58384228a0f05

RUN export NVM_DIR=/root/.nvm && \
    source $NVM_DIR/nvm.sh && \
    nvm use 18 && \
    node -v && \
    BUILD_OHIF=true pip install -e MONAILabel

RUN cd /opt/holoscan/monailabel && monailabel apps --download --name monaibundle --output .

RUN pip install pydicom==2.4.4 nilearn

RUN mkdir -p /opt/holoscan/monailabel/monaibundle/model
RUN cd /opt/holoscan/monailabel/monaibundle/model && wget https://raw.githubusercontent.com/SimoneBendazzoli93/MONet-Bundle/main/MONetBundle.zip
RUN unzip /opt/holoscan/monailabel/monaibundle/model/MONetBundle.zip -d /opt/holoscan/monailabel/monaibundle/model/ 

ENTRYPOINT ["/var/holoscan/tools"]