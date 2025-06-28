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

from pydicom.sr.codedict import codes

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from pydicom.sr.codedict import Code
from monai.deploy.core.domain import Image
from monai.deploy.core.io_type import IOType
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.monai_bundle_inference_operator import BundleConfigNames, IOMapping
from monai.deploy.operators.monet_bundle_inference_operator import MONetBundleInferenceOperator
from monai.deploy.operators.stl_conversion_operator import STLConversionOperator


# @resource(cpu=1, gpu=1, memory="7Gi")
# pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# The monai pkg is not required by this class, instead by the included operators.
class AILymphomaMONetSegApp(Application):

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
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="study_loader_op"
        )
        CT_series_selector_op = DICOMSeriesSelectorOperator(self, rules=CT_Rules_Text, name="ct_series_selector_op")
        CT_series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="ct_series_to_vol_op")
        
        PT_series_selector_op = DICOMSeriesSelectorOperator(self, rules=PET_Rules_Text, name="pt_series_selector_op")
        PT_series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="pt_series_to_vol_op")


        config_names = BundleConfigNames(config_names=["inference"])  # Same as the default

        bundle_lymphoma_seg_op = MONetBundleInferenceOperator(
            self,
            input_mapping=[IOMapping("CT", Image, IOType.IN_MEMORY),IOMapping("PT", Image, IOType.IN_MEMORY)],
            output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
            app_context=app_context,
            bundle_config_names=config_names,
            ref_modality="PT",  # Use the PET modality as the reference for segmentation
            name="monet_bundle_lymphoma_seg_op",
        )

        segment_descriptions = [
            SegmentDescription(
                segment_label="Lesion",
                segmented_property_category = codes.SCT.Lesion,
                segmented_property_type = Code("255081000000109", "SCT", "Lymphoma"),   
                algorithm_name="volumetric (3D) segmentation of Lymphoma Lesion from PET-CT image",
                algorithm_family=codes.DCM.ArtificialIntelligence,
                algorithm_version="0.3.2",
            )
        ]

        custom_tags = {"SeriesDescription": "AI generated Seg, not for clinical use."}

        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            custom_tags=custom_tags,
            output_folder=app_output_path,
            name="dicom_seg_writer",
        )

        self.add_flow(study_loader_op, CT_series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            CT_series_selector_op, CT_series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(CT_series_to_vol_op, bundle_lymphoma_seg_op, {("image", "CT")})
        
        
        self.add_flow(study_loader_op, PT_series_selector_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(
            PT_series_selector_op, PT_series_to_vol_op, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(PT_series_to_vol_op, bundle_lymphoma_seg_op, {("image", "PT")})
        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        self.add_flow(
            PT_series_selector_op, dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
        )
        self.add_flow(bundle_lymphoma_seg_op, dicom_seg_writer, {("pred", "seg_image")})

        stl_conversion_op = STLConversionOperator(
            self, output_file=app_output_path.joinpath("stl/Lymphoma_Lesions.stl"), name="stl_conversion_op"
        )
        self.add_flow(bundle_lymphoma_seg_op, stl_conversion_op, {("pred", "image")})

        logging.info(f"End {self.compose.__name__}")



CT_Rules_Text = """
{
    "selections": [
        {
            "name": "CT Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)CT",
                "SeriesDescription": "(.*?)"
            }
        }
    ]
}
"""

PET_Rules_Text = """
{
    "selections": [
        {
            "name": "PET Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)PT",
                "SeriesDescription": "(.*?)"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    AILymphomaMONetSegApp().run()
    logging.info(f"End {__name__}")
