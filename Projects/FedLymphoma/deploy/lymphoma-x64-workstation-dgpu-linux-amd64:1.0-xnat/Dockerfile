
FROM maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64:1.0
USER root

COPY ./tools /var/holoscan/tools
RUN chmod +x /var/holoscan/tools

# Set the working directory
WORKDIR /var/holoscan



ENV PATH=/root/.local/bin:/opt/nvidia/holoscan/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/lib/python3.10/site-packages/holoscan/lib

COPY ./pip/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt


# Install MONAI Deploy App SDK

# Install MONAI Deploy from PyPI org
# RUN pip install monai-deploy-app-sdk==2.0.0
RUN pip install --user git+https://github.com/SimoneBendazzoli93/monai-deploy-app-sdk.git@75af6f32c10c37c849e32d2c55ff8845699da1a1
COPY ./map/app.json /etc/holoscan/app.json
COPY ./app.config /var/holoscan/app.yaml
COPY ./map/pkg.json /etc/holoscan/pkg.json

COPY ./app /opt/holoscan/app

COPY ./models  /opt/holoscan/models
LABEL org.nrg.commands="[{\"name\": \"lymphoma-segmentation\", \"label\": \"lymphoma-segmentation\", \"description\": \"Runs Lymphoma Segmentation on a session derived PET-CT scan.\", \"version\": \"1.0\", \"schema-version\": \"1.0\", \"info-url\": \"https://github.com/rordenlab/dcm2niix\", \"image\": \"maiacloud/lymphoma-x64-workstation-dgpu-linux-amd64:1.0-xnat\", \"type\": \"docker\", \"command-line\": \"\", \"mounts\": [{\"name\": \"dicom-in\", \"writable\": true, \"path\": \"/var/holoscan/input\"}, {\"name\": \"dicom-out\", \"writable\": true, \"path\": \"/var/holoscan/output\"}], \"environment-variables\": {\"ORTHANC_URL\": \"orthancURL\"}, \"ports\": {}, \"inputs\": [{\"name\": \"orthanc-url\", \"description\": \"Orthanc URL\", \"type\": \"string\", \"required\": true, \"replacement-key\": \"orthancURL\", \"select-values\": []}], \"outputs\": [{\"name\": \"dicom-seg\", \"description\": \"The output segmentation\", \"required\": true, \"mount\": \"dicom-out\"}], \"xnat\": [{\"name\": \"spleen-segmentation\", \"label\": \"spleen-segmentation\", \"description\": \"Runs Spleen Segmentation on a session derived scan.\", \"contexts\": [\"xnat:imageSessionData\"], \"external-inputs\": [{\"name\": \"session\", \"description\": \"Input session\", \"type\": \"Session\", \"matcher\": \"'DICOM' in @.scans[*].resources[*].label\", \"required\": true, \"load-children\": true}], \"derived-inputs\": [{\"name\": \"scan\", \"description\": \"Input scan\", \"type\": \"Scan\", \"matcher\": \"'DICOM' in @.resources[*].label\", \"required\": true, \"load-children\": true, \"derived-from-wrapper-input\": \"session\", \"multiple\": false}, {\"name\": \"scan-dicoms\", \"description\": \"The dicom resource on the scan\", \"type\": \"Resource\", \"matcher\": \"@.label == 'DICOM'\", \"required\": true, \"provides-files-for-command-mount\": \"dicom-in\", \"load-children\": true, \"derived-from-wrapper-input\": \"scan\", \"multiple\": false}], \"output-handlers\": [{\"name\": \"dicom-seg-resource\", \"accepts-command-output\": \"dicom-seg\", \"as-a-child-of\": \"scan\", \"type\": \"Resource\", \"label\": \"DICOM-SEG\", \"format\": \"DICOM\", \"tags\": []}]}], \"shm-size\": 1073741824, \"container-labels\": {}, \"generic-resources\": {\"nvidia.com/gpu\": \"1\"}, \"ulimits\": {}, \"secrets\": []}]"

ENTRYPOINT ["/var/holoscan/tools"]