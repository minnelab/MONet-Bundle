
ARG GPU_TYPE=dgpu




FROM nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
        curl \
        jq \
        git \
    && rm -rf /var/lib/apt/lists/*




# FROM base AS mofed-installer
# ARG MOFED_VERSION=23.10-2.1.3.1

# # In a container, we only need to install the user space libraries, though the drivers are still
# # needed on the host.
# # Note: MOFED's installation is not easily portable, so we can't copy the output of this stage
# # to our final stage, but must inherit from it. For that reason, we keep track of the build/install
# # only dependencies in the `MOFED_DEPS` variable (parsing the output of `--check-deps-only`) to
# # remove them in that same layer, to ensure they are not propagated in the final image.
# WORKDIR /opt/nvidia/mofed
# ARG MOFED_INSTALL_FLAGS="--dpdk --with-mft --user-space-only --force --without-fw-update"
# RUN UBUNTU_VERSION=$(cat /etc/lsb-release | grep DISTRIB_RELEASE | cut -d= -f2) \
#     && OFED_PACKAGE="MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu${UBUNTU_VERSION}-$(uname -m)" \
#     && curl -S -# -o ${OFED_PACKAGE}.tgz -L \
#         https://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VERSION}/${OFED_PACKAGE}.tgz \
#     && tar xf ${OFED_PACKAGE}.tgz \
#     && MOFED_INSTALLER=$(find . -name mlnxofedinstall -type f -executable -print) \
#     && MOFED_DEPS=$(${MOFED_INSTALLER} ${MOFED_INSTALL_FLAGS} --check-deps-only 2>/dev/null | tail -n1 |  cut -d' ' -f3-) \
#     && apt-get update \
#     && apt-get install --no-install-recommends -y ${MOFED_DEPS} \
#     && ${MOFED_INSTALLER} ${MOFED_INSTALL_FLAGS} \
#     && rm -r * \
#     && apt-get remove -y ${MOFED_DEPS} && apt-get autoremove -y \
#     && rm -rf /var/lib/apt/lists/*

FROM base AS release
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

ARG GPU_TYPE
ARG UNAME
ARG UID
ARG GID

RUN mkdir -p /etc/holoscan/ \
        && mkdir -p /opt/holoscan/ \
        && mkdir -p /var/holoscan \
        && mkdir -p /opt/holoscan/app \
        && mkdir -p /var/holoscan/input \
        && mkdir -p /var/holoscan/output

LABEL base="nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu22.04"
LABEL tag="spleen:1.0"
LABEL org.opencontainers.image.title="MONAI Deploy App Package - Spleen nnUNet Seg Inference"
LABEL org.opencontainers.image.version="1.0"
LABEL org.nvidia.holoscan="3.1.0"

LABEL org.monai.deploy.app-sdk="2.0.0"

ENV HOLOSCAN_INPUT_PATH=/var/holoscan/input
ENV HOLOSCAN_OUTPUT_PATH=/var/holoscan/output
ENV HOLOSCAN_WORKDIR=/var/holoscan
ENV HOLOSCAN_APPLICATION=/opt/holoscan/app
ENV HOLOSCAN_TIMEOUT=0
ENV HOLOSCAN_MODEL_PATH=/opt/holoscan/models
ENV HOLOSCAN_DOCS_PATH=/opt/holoscan/docs
ENV HOLOSCAN_CONFIG_PATH=/var/holoscan/app.yaml
ENV HOLOSCAN_APP_MANIFEST_PATH=/etc/holoscan/app.json
ENV HOLOSCAN_PKG_MANIFEST_PATH=/etc/holoscan/pkg.json
ENV HOLOSCAN_LOGS_PATH=/var/holoscan/logs
ENV HOLOSCAN_VERSION=3.1.0









# If torch is installed, we can skip installing Python
ENV PYTHON_VERSION=3.10.6-1~22.04
ENV PYTHON_PIP_VERSION=22.0.2+dfsg-*

RUN apt update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
        python3-minimal=${PYTHON_VERSION} \
        libpython3-stdlib=${PYTHON_VERSION} \
        python3=${PYTHON_VERSION} \
        python3-venv=${PYTHON_VERSION} \
        python3-pip=${PYTHON_PIP_VERSION} \
    && rm -rf /var/lib/apt/lists/*







RUN groupadd -f -g $GID $UNAME
RUN useradd -rm -d /home/$UNAME -s /bin/bash -g $GID -G sudo -u $UID $UNAME
RUN chown -R holoscan /var/holoscan && \
    chown -R holoscan /var/holoscan/input && \
    chown -R holoscan /var/holoscan/output

# Set the working directory
WORKDIR /var/holoscan

# Copy HAP/MAP tool script
COPY ./tools /var/holoscan/tools
RUN chmod +x /var/holoscan/tools

# Set the working directory
WORKDIR /var/holoscan

USER $UNAME

ENV PATH=/home/${UNAME}/.local/bin:/opt/nvidia/holoscan/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/${UNAME}/.local/lib/python3.10/site-packages/holoscan/lib

COPY ./pip/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt


# Install MONAI Deploy App SDK

# Install MONAI Deploy from PyPI org
# RUN pip install monai-deploy-app-sdk==2.0.0
RUN pip install git+https://github.com/SimoneBendazzoli93/monai-deploy-app-sdk.git

COPY ./models  /opt/holoscan/models


COPY ./map/app.json /etc/holoscan/app.json
COPY ./app.config /var/holoscan/app.yaml
COPY ./map/pkg.json /etc/holoscan/pkg.json

COPY ./app /opt/holoscan/app


ENTRYPOINT ["/var/holoscan/tools"]