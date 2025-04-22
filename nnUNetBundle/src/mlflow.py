
import re
from monai.handlers import  MLFlowHandler
import yaml
from monai.bundle import ConfigParser
from pathlib import Path
import monai
import torch

def mlflow_transform(state_output):
    """
    Extracts the 'loss' value from the first element of the state_output list.

    Parameters
    ----------
    state_output : list of dict
        A list where each element is a dictionary containing various metrics, including 'loss'.

    Returns
    -------
    float
        The 'loss' value from the first element of the state_output list.
    """
    return state_output[0]['loss']

class MLFlownnUNetHandler(MLFlowHandler):
    """
    A handler for logging nnUNet metrics to MLFlow.
    Parameters
    ----------
    label_dict : dict
        A dictionary mapping label indices to label names.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.
    """
    def __init__(self, label_dict, **kwargs):
        super(MLFlownnUNetHandler, self).__init__(**kwargs)
        self.label_dict = label_dict
        
    def _default_epoch_log(self, engine) -> None:
        """
        Logs the metrics and state attributes at the end of each epoch.

        Parameters
        ----------
        engine : Engine
            The engine object that contains the state and metrics to be logged.

        Returns
        -------
        None
        """
        log_dict = engine.state.metrics
        if not log_dict:
            return

        current_epoch = self.global_epoch_transform(engine.state.epoch)

        new_log_dict = {}

        for metric in log_dict:
            if type(log_dict[metric]) == torch.Tensor:
                for i,val in enumerate(log_dict[metric]):
                    new_log_dict[metric+"_{}".format(list(self.label_dict.keys())[i+1])] = val
            else:
                new_log_dict[metric] = log_dict[metric]
        self._log_metrics(new_log_dict, step=current_epoch)

        if self.state_attributes is not None:
            attrs = {attr: getattr(engine.state, attr, None) for attr in self.state_attributes}
            self._log_metrics(attrs, step=current_epoch)

def create_mlflow_experiment_params(params_file, custom_params=None):
    """
    Create a dictionary of parameters for an MLflow experiment.

    This function reads configuration values from MONAI, GPU information, and a YAML configuration file,
    and combines them into a single dictionary. Optionally, custom parameters can also be added to the dictionary.

    Parameters
    ----------
    params_file : str
        Path to the YAML configuration file.
    custom_params : dict, optional
        A dictionary of custom parameters to be added to the final parameters dictionary (default is None).

    Returns
    -------
    dict
        A dictionary containing all the combined parameters.
    """
    params_dict = {}
    config_values = monai.config.deviceconfig.get_config_values()
    for k in config_values:
        params_dict[re.sub("[()]"," ",str(k))] = config_values[k]

    optional_config_values = monai.config.deviceconfig.get_optional_config_values()
    for k in optional_config_values:
        params_dict[re.sub("[()]"," ",str(k))] = optional_config_values[k]

    gpu_info = monai.config.deviceconfig.get_gpu_info()
    for k in gpu_info:
        params_dict[re.sub("[()]"," ",str(k))] = str(gpu_info[k])

    yaml_config_files = [params_file]
    # %%
    monai_config = {}
    for config_file in yaml_config_files:
        with open(config_file, 'r') as file:
            monai_config.update(yaml.safe_load(file))

    monai_config["bundle_root"] = str(Path(Path(params_file).parent).parent)

    parser = ConfigParser(monai_config, globals={"os": "os",
                                                 "pathlib": "pathlib",
                                                 "json": "json",
                                                 "ignite": "ignite"
                                                 })

    parser.parse(True)

    for k in monai_config:
        params_dict[k] = parser.get_parsed_content(k,instantiate=True)

    if custom_params is not None:
        for k in custom_params:
            params_dict[k] = custom_params[k]
    return params_dict
