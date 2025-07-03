# MONet Local Inference

If you want to run MONet inference locally, without using the provided Docker images, you can do so by following these instructions, provided to help you set up the environment and run the inference script.

## Requirements
- Python 3.10 or higher
- NVidia GPU with CUDA support
- PyTorch 2.4.1 with CUDA support

Recommended pytorch installation command:
```bash
pip install light-the-torch
ltt install torch==2.4.1 torchvision
```
- Other dependencies can be installed using the provided `requirements.txt` file.
```bash
pip install -r requirements.txt
```
## Download Model Locally
To download the pre-trained MONet model locally, you can use the following command:
```bash
wget https://<MONAI-LABEL_SERVER_URL>/model/MONetBundle -O <MODEL_NAME>.ts
```

## Run Inference
To run the MONet inference script, you can use the following command:
```bash
python predict.py --input-image <INPUT_IMAGE> --output-folder <OUTPUT_DIR> --model <MODEL_NAME>.ts
```

