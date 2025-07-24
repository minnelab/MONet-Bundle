import os

import setuptools
from setuptools import setup

import versioneer


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


setup(
    version=versioneer.get_version(),
    packages=setuptools.find_packages(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    data_files=[('', ["requirements.txt"]), ],
    package_data={
        "": ["configs/*.yaml", "configs/*.json", "configs/*.yml"],
        "MONet.icons": ["*.png", "*.svg"],
    },
    entry_points={
        "console_scripts": [
            "MONet_concatenate_modalities = MONet_scripts.MONet_concatenate_modalities:main",
            "MONet_convert_ckpt_to_ts = MONet_scripts.MONet_convert_ckpt_to_ts:main",
            "MONet_login = MONet_scripts.MONet_login:main",
            "MONet_local_inference = MONet_scripts.MONet_local_inference:main",
            "MONet_convert_nnunet_checkpoint_to_bundle_checkpoint = MONet_scripts.MONet_convert_nnunet_checkpoint_to_bundle_checkpoint:main",
            "MONet_fetch_bundle = MONet_scripts.MONet_fetch_bundle:main",
            "MONet_remote_inference = MONet_scripts.MONet_remote_inference:main",
            "MONet_run_conversion = MONet_scripts.MONet_run_conversion:main",
            "MONet_inference_nifti = MONet_scripts.MONet_inference_nifti:main",
            "MONet_inference_dicom = MONet_scripts.MONet_inference_dicom:main",
            "MAIA-Segmentation-Portal = MONet_scripts.MAIA_Segmentation_Portal:main",
            "MONet_MONAI_Label = MONet_scripts.MONet_MONAI_Label:main",
            "MONet_pipeline = MONet_scripts.MONet_pipeline:main",
            "MONet_PL = MONet_scripts.MONet_PL:main",
        ],
    },
    keywords=["monai", "model inference", "pytorch", "monet bundle", "maia"],

)
