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
from pydicom.sr.codedict import Code

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.core.domain import Image
from monai.deploy.core.io_type import IOType
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.monai_bundle_inference_operator import BundleConfigNames, IOMapping
from monai.deploy.operators.monet_bundle_inference_operator import MONetBundleInferenceOperator
from monai.deploy.operators.stl_conversion_operator import STLConversionOperator
import os
import yaml

# @resource(cpu=1, gpu=1, memory="7Gi")
# pip_packages can be a string that is a path(str) to requirements.txt file or a list of packages.
# The monai pkg is not required by this class, instead by the included operators.
class AIMONetSegApp(Application):
    
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
        
        segmentation_task_params = None
        with open(os.environ["SEGMENTATION_TASK_CONFIG_FILE"],"r") as f:
            segmentation_task_params = yaml.safe_load(f)["tasks"]

        input_mapping = []
        ref_modality = "image"
        series_selector_ops = {}
        series_to_vol_ops = {}
        for modality in segmentation_task_params[os.environ["SEGMENTATION_TASK_NAME"]]["Modalities"]:
            series_selector_ops[modality] = DICOMSeriesSelectorOperator(self, rules=segmentation_task_params[os.environ["SEGMENTATION_TASK_NAME"]]["Modalities"][modality]["Rules_Text"], name=f"{modality.lower()}_series_selector_op")
            series_to_vol_ops[modality] = DICOMSeriesToVolumeOperator(self, name=f"{modality.lower()}_series_to_vol_op")
            input_mapping.append(IOMapping(modality, Image, IOType.IN_MEMORY))

        if "Reference_Modality" in segmentation_task_params[os.environ["SEGMENTATION_TASK_NAME"]]:
            ref_modality = segmentation_task_params[os.environ["SEGMENTATION_TASK_NAME"]]["Reference_Modality"]


        config_names = BundleConfigNames(config_names=["inference"])  # Same as the default

        bundle_seg_op = MONetBundleInferenceOperator(
            self,
            input_mapping=input_mapping,
            output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
            app_context=app_context,
            ref_modality=ref_modality,
            bundle_config_names=config_names,
            name="monet_bundle_seg_op",
        )

        # Create DICOM Seg writer providing the required segment description for each segment with
        # the actual algorithm and the pertinent organ/tissue. The segment_label, algorithm_name,
        # and algorithm_version are of DICOM VR LO type, limited to 64 chars.
        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
        segment_descriptions = []
        
        for segment in segmentation_task_params[os.environ["SEGMENTATION_TASK_NAME"]]["Segments"]:
            segment_descriptions.append(
                SegmentDescription(
                    segment_label=segment["segment_label"],
                    segmented_property_category=Code(segment["segmented_property_category"]["code"], segment["segmented_property_category"]["scheme"], segment["segmented_property_category"]["meaning"]),
                    segmented_property_type=Code(segment["segmented_property_type"]["code"], segment["segmented_property_type"]["scheme"], segment["segmented_property_type"]["meaning"]),
                algorithm_name=segment["algorithm_name"],
                algorithm_family=codes.DCM.ArtificialIntelligence,
                algorithm_version="0.3.2",
            ),
            )
            

        custom_tags = {"SeriesDescription": "AI generated Seg, not for clinical use."}

        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            custom_tags=custom_tags,
            output_folder=app_output_path,
            name="dicom_seg_writer",
        )
        
        

        for modality in segmentation_task_params[os.environ["SEGMENTATION_TASK_NAME"]]["Modalities"]:
            self.add_flow(study_loader_op, series_selector_ops[modality], {("dicom_study_list", "dicom_study_list")})
            self.add_flow(series_selector_ops[modality], series_to_vol_ops[modality], {("study_selected_series_list", "study_selected_series_list")})
            self.add_flow(series_to_vol_ops[modality], bundle_seg_op, {("image", modality)})
          

        if ref_modality != "image":
            self.add_flow(
                series_selector_ops[ref_modality], dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
            )
        else:
            self.add_flow(
                series_selector_ops["image"], dicom_seg_writer, {("study_selected_series_list", "study_selected_series_list")}
            )
        self.add_flow(bundle_seg_op, dicom_seg_writer, {("pred", "seg_image")})
        # Create the surface mesh STL conversion operator and add it to the app execution flow, if needed, by
        # uncommenting the following couple lines.
        #stl_conversion_op = STLConversionOperator(
        #    self, output_file=app_output_path.joinpath("stl/brain_lesion.stl"), name="stl_conversion_op"
        #)
        #self.add_flow(bundle_brats_seg_op, stl_conversion_op, {("pred", "image")})

        logging.info(f"End {self.compose.__name__}")
        

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    AIMONetSegApp().run()
    logging.info(f"End {__name__}")
