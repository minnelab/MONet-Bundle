from __future__ import annotations

import requests
from loguru import logger

def get_available_models(token, username):
    """
    Fetches available segmentation models for a given user from the remote portal.

    Parameters
    ----------
    token : str
        The authentication token for the user.
    username : str
        The username of the user.

    Returns
    -------
    dict
        A dictionary mapping model names (without '-segmentation' suffix) to their corresponding MONAI label.
    """
    try:
        response = requests.post(
            "https://maia.app.cloud.cbh.kth.se/maia/maia-segmentation-portal/models/",
            data={"id_token": token, "username": username},
        )
    except requests.RequestException as e:
        logger.error(f"Error fetching models: {e}")
        return {}
    model_list = response.json()["models"]

    models = {}
    for model in model_list:
        model_name = model
        if model_name.endswith("-segmentation"):
            model_name = model_name[: -len("-segmentation")]
        models[model_name] = model_list[model]["monai_label"]

    return models
