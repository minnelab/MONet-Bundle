# MONet Bundle

The MONet Bundle is a MONAI Bundle designed to wrap nnUNet for use in MONAI applications. It provides a convenient way to integrate nnUNet's capabilities into the MONAI framework, allowing users to bring the power of nnUNet to their MONAI workflows.
The MONet Bundle can be used to train and perform inference with nnUNet models.

## Installation
To install the MONet Bundle, you can use the following command:

```bash
wget https://raw.githubusercontent.com/SimoneBendazzoli93/MONet-Bundle/main/MONetBundle.zip
unzip MONetBundle.zip
```


### Training

To train a model using the MONet Bundle, you need to have the nnUNet preprocessed data available, together with the corresponding nnUNet plan. You can use the native `nnUNet_plan_and_preprocess` command to preprocess your data. In alternative, you can use the `nnUNetRunnerV2` class from MONAI to handle the preprocessing step.

```python
runner = nnUNetV2Runner(
    input_config=data_src_cfg, trainer_class_name="nnUNetTrainer", work_dir=nnunet_root_dir
)
runner.convert_dataset()
runner.plan_and_process()
```
After preprocessing, you can train the nnUNet model with the MONet Bundle:

```bash
export nnUNet_results=/home/maia-user/Data/nnUNet/nnUNet_trained_models
export nnUNet_raw=/home/maia-user/Data/nnUNet/nnUNet_raw_data_base
export nnUNet_preprocessed=/home/maia-user/Data/nnUNet/nnUNet_preprocessed
export nnUNet_def_n_proc=2
export nnUNet_n_proc_DA=2
export BUNDLE_ROOT=MONetBundle
export PYTHONPATH=$PYTHONPATH:$BUNDLE_ROOT

python -m monai.bundle run \
    --bundle_root $BUNDLE_ROOT \
    --dataset_name_or_id '09' \
    --fold_id 0 \
    --iterations 10 \                           # OPTIONAL: number of iterations per epoch, default is 250
    --epochs 10 \                               # OPTIONAL: number of epochs, default is 1000
    --label_dict.0 background \                 # OPTIONAL: label dictionary, default is {0: 'background', 1: 'class1'}
    --label_dict.1 Spleen \                     # OPTIONAL: label dictionary, default is {0: 'background', 1: 'class1'}
    --mlflow_experiment_name Task09_Spleen \    # OPTIONAL: name of the MLflow experiment, default is 'MONetBundle'
    --mlflow_run_name Task09_Spleen_fold0 \     # OPTIONAL: name of the MLflow run, default is 'MONetBundle'
    --tracking_uri  http://localhost:5000 \     # OPTIONAL: MLflow tracking URI, default is 'mlruns'
    --nnunet_plans_identifier nnUNetPlans \     # OPTIONAL: identifier for the nnUNet plans, default is 'nnUNetPlans'
    --nnunet_trainer_class_name nnUNetTrainer \ # OPTIONAL: name of the nnUNet trainer class, default is 'nnUNetTrainer'
    --num_classes 2 \                           # OPTIONAL: number of classes, default is 2
    --nnunet_configuration 3d_fullres \         # OPTIONAL: nnUNet configuration, default is '3d_fullres'
    --config_file $BUNDLE_ROOT/configs/train.yaml
```
To resume the training from a previous checkpoint, you can use the `--reload_checkpoint_epoch "latest"` flag and specify the config files:

```bash
python -m monai.bundle run \
    ... (other parameters) \
    --reload_checkpoint_epoch "latest" \
    --config_file "['$BUNDLE_ROOT/configs/train.yaml','$BUNDLE_ROOT/configs/train_resume.yaml']"
```

### Inference
To perform inference with the trained nnUNet model, you can use the following command:

```bash
python -m monai.bundle run \
    --config-file $BUNDLE_ROOT/configs/inference.yaml \
    --bundle-root $BUNDLE_ROOT \
    --data-dir $HOME/Data/Task09_Spleen \
    --output-dir $HOME/Data/Task09_Spleen \
    --model-name "checkpoint_epoch=10.pt" \
    --fold_id 0 \
    --logging-file $BUNDLE_ROOT/configs/logging.conf
```

The input data is organized in subfolders, where each subfolder corresponds to a subject and contains the image for that subject.
```plaintext
  [Dataset_folder]
        [Subject_0]
            - Subject_0_image.nii.gz    # Subject_0 modality 0
        [Subject_1]
            - Subject_1_image.nii.gz    # Subject_1 modality 0
```
The output will be saved in the specified output directory, with the same structure as the input data, but with the predicted labels in NIfTI format.
