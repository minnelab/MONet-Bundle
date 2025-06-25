# Copyright 2021-2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.core.domain import Image
from monai.deploy.core.io_type import IOType
from monai.deploy.operators.monai_bundle_inference_operator import BundleConfigNames, IOMapping
from monai.deploy.operators.monet_bundle_inference_operator import MONetBundleInferenceOperator
from monai.deploy.operators.nii_data_loader_operator import NiftiDataLoader
from monai.deploy.operators.nii_data_writer_operator import NiftiDataWriter


# @resource(cpu=1, gpu=1, memory="7Gi")
# pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# The monai pkg is not required by this class, instead by the included operators.
class AILymphomaNIFTIMONetSegApp(Application):

    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        logging.info(f"Begin {self.compose.__name__}")

        # Use Commandline options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)

        # Create the custom operator(s) as well as SDK built-in operator(s).
        ct_nifti_loader_op = NiftiDataLoader(
            self, CountCondition(self, 1), input_path=app_input_path, name="ct_nifti_loader_op", modality_mapping="_ct.nii.gz"
        )

        pt_nifti_loader_op = NiftiDataLoader(
            self, CountCondition(self, 1), input_path=app_input_path, name="pt_nifti_loader_op", modality_mapping="_pet.nii.gz"
        )
        # Create the inference operator that supports MONAI Bundle and automates the inference.
        # The IOMapping labels match the input and prediction keys in the pre and post processing.
        # The model_name is optional when the app has only one model.
        # The bundle_path argument optionally can be set to an accessible bundle file path in the dev
        # environment, so when the app is packaged into a MAP, the operator can complete the bundle parsing
        # during init.

        config_names = BundleConfigNames(config_names=["inference"])  # Same as the default

        bundle_spleen_seg_op = MONetBundleInferenceOperator(
            self,
            input_mapping=[IOMapping("CT", Image, IOType.IN_MEMORY),IOMapping("PT", Image, IOType.IN_MEMORY)],
            output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
            app_context=app_context,
            bundle_config_names=config_names,
            name="nnunet_bundle_spleen_seg_op",
        )

        # Create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue. The segment_label, algorithm_name,
        # and algorithm_version are of DICOM VR LO type, limited to 64 chars.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html

        output_file = Path(app_output_path).joinpath(Path(app_input_path).name+".nii.gz")

        if Path(output_file).is_file():
            output_file = Path(app_output_path).joinpath(Path(app_input_path).name+"_seg.nii.gz")

        nifti_seg_writer = NiftiDataWriter(
            self,
            output_file=output_file,
            name="nifti_seg_writer",
        )

        # Create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type.
        self.add_flow(ct_nifti_loader_op, bundle_spleen_seg_op, {("image", "CT")})
        self.add_flow(pt_nifti_loader_op, bundle_spleen_seg_op, {("image", "PT")})
        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        self.add_flow(bundle_spleen_seg_op, nifti_seg_writer, {("pred", "seg_image")})
        # Create the surface mesh STL conversion operator and add it to the app execution flow, if needed, by
        logging.info(f"End {self.compose.__name__}")


if __name__ == "__main__":

    logging.info(f"Begin {__name__}")
    AILymphomaNIFTIMONetSegApp().run()
    logging.info(f"End {__name__}")
