from __future__ import annotations

# debug_logger()
import argparse
from pathlib import Path
from time import sleep

import pydicom
from pydicom.dataset import Dataset
from pynetdicom import AE, StoragePresentationContexts, build_role, evt
from pynetdicom.apps.storescu.storescu import main as storescu_main
from pynetdicom.sop_class import (
    CTImageStorage,
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelGet,
    PositronEmissionTomographyImageStorage,
    SegmentationStorage,
)
from tqdm import tqdm


def get_arg_parser():
    """Get the argument parser."""

    parser = argparse.ArgumentParser(description="Find and retrieve DICOM Images from PACS.")
    parser.add_argument("--pacs-ip", type=str, required=True, help="IP address of the PACS server.")
    parser.add_argument("--pacs-port", type=int, default=4242, help="Port of the PACS server (default: 4242).")
    parser.add_argument("--calling-ae-title", type=str, required=True, help="AE Title of the calling application entity.")
    parser.add_argument("--called-ae-title", type=str, required=True, help="AE Title of the called application entity (PACS).")
    parser.add_argument("--input-folder", type=str, required=True, help="Path to the input folder containing DICOM files.")
    parser.add_argument(
        "--output-folder",
        type=str,
        required=False,
        help="Path to the output folder where retrieved DICOM images will be saved. Required unless --send-only is set.",
    )
    parser.add_argument("--send-only", action="store_true", help="If set, only send the DICOM files without retrieving them.")
    args = parser.parse_args()
    return args


def query(ae, PACS_IP, PACS_PORT, CALLED_AE_TITLE, query):
    # Create our Identifier (query) dataset
    ds = Dataset()

    ds.PatientName = ""
    ds.PatientID = ""
    ds.StudyID = ""
    ds.AccessionNumber = ""
    ds.QueryRetrieveLevel = "SERIES"
    ds.SOPClassesInStudy = ""
    ds.StudyInstanceUID = ""
    ds.SeriesInstanceUID = ""
    ds.NumberOfSeriesRelatedInstances = ""
    ds.Modality = ""
    ds.SeriesDescription = ""

    for key in query:
        if key in ds:
            ds.data_element(key).value = query[key]

    study_instance_uid_list = []
    # Associate with the peer AE at PACS IP and port
    assoc = ae.associate(PACS_IP, PACS_PORT, ae_title=CALLED_AE_TITLE)
    if assoc.is_established:
        # Send the C-FIND request
        responses = assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind)
        for status, identifier in responses:
            if status:
                # print(identifier.PatientName)
                # print(identifier.StudyID)
                # if identifier:
                #    print(identifier.SeriesDescription)
                if identifier and identifier.StudyInstanceUID not in study_instance_uid_list:

                    study_instance_uid_list.append(identifier.StudyInstanceUID)

                print("C-FIND query status: 0x{0:04X}".format(status.Status))
            else:
                print("Connection timed out, was aborted or received invalid response")

        # Release the association
        assoc.release()
    else:
        print("Association rejected, aborted or never connected")

    print("Study Instance UIDs found:")
    for uid in study_instance_uid_list:
        print(uid)
    return study_instance_uid_list


def retrieve(ae, PACS_IP, PACS_PORT, CALLED_AE_TITLE, study_instance_uid_list, ROOT_FOLDER):
    ae.requested_contexts = StoragePresentationContexts[:127]
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)

    for study_instance_uid in tqdm(study_instance_uid_list):
        ds = Dataset()
        # ds.PatientName = ''
        # ds.PatientID = ''
        # ds.StudyID = ''
        # ds.AccessionNumber = ''
        ds.QueryRetrieveLevel = "STUDY"
        # ds.SOPClassesInStudy = ''
        ds.StudyInstanceUID = study_instance_uid
        # ds.SeriesInstanceUID = ''
        # ds.NumberOfSeriesRelatedInstances = ''
        ds.Modality = "SEG"
        # ds.SeriesDescription = ''

        roles = []
        for cx in [CTImageStorage, PositronEmissionTomographyImageStorage, SegmentationStorage]:
            roles.append(build_role(cx, scp_role=True))

        # debug_logger()
        # Implement the handler for evt.EVT_C_STORE
        def handle_store(event):
            """Handle a C-STORE request event."""
            ds = event.dataset
            ds.file_meta = event.file_meta
            print("Received C-STORE request for SOP Instance UID: {}".format(ds.SOPInstanceUID))
            Path(ROOT_FOLDER).joinpath(ds.PatientID, ds.StudyDescription + "-" + ds.StudyInstanceUID, ds.SeriesDescription).mkdir(
                parents=True, exist_ok=True
            )
            # Save the dataset using the SOP Instance UID as the filename
            ds.save_as(
                Path(ROOT_FOLDER).joinpath(
                    ds.PatientID, ds.StudyDescription + "-" + ds.StudyInstanceUID, ds.SeriesDescription, ds.SOPInstanceUID
                ),
                write_like_original=False,
            )
            print(
                Path(ROOT_FOLDER).joinpath(
                    ds.PatientID, ds.StudyDescription + "-" + ds.StudyInstanceUID, ds.SeriesDescription, ds.SOPInstanceUID
                )
            )
            # Return a 'Success' status
            return 0x0000

        handlers = [(evt.EVT_C_STORE, handle_store)]

        assoc = ae.associate(PACS_IP, PACS_PORT, ext_neg=roles, evt_handlers=handlers, ae_title=CALLED_AE_TITLE)

        if assoc.is_established:
            # Use the C-GET service to send the identifier
            responses = assoc.send_c_get(ds, PatientRootQueryRetrieveInformationModelGet)
            for status, identifier in responses:
                if status:
                    print("C-GET query status: 0x{0:04x}".format(status.Status))
                else:
                    print("Connection timed out, was aborted or received invalid response")

            # Release the association
            assoc.release()
        else:
            print("Association rejected, aborted or never connected")


def main():
    args = get_arg_parser()
    # Define PACS connection details
    PACS_IP = args.pacs_ip  # Replace with actual PACS IP
    PACS_PORT = args.pacs_port  # Replace with actual PACS port
    CALLING_AE_TITLE = args.calling_ae_title  # Your AE Title
    CALLED_AE_TITLE = args.called_ae_title  # PACS AE Title
    ae = AE(ae_title=CALLING_AE_TITLE)

    ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    storescu_main(["storescu", "-r", PACS_IP, str(PACS_PORT), input_folder])

    if args.send_only:
        print("Send-only flag is set. Exiting after sending DICOM files.")
        return

    # Find the first .dcm file in input_folder (including subdirectories)
    first_dcm_file = None
    for dcm_path in input_folder.rglob("*.dcm"):
        first_dcm_file = dcm_path
        break

    if first_dcm_file is None:
        print("No .dcm file found in the input folder.")
        return

    # Example: extract PatientID from the first DICOM file
    ds = pydicom.dcmread(first_dcm_file)
    patient_id = ds.get("PatientID", "")

    query_dict = {"Modality": "SEG", "PatientID": patient_id}

    study_instance_uid_list = []
    while len(study_instance_uid_list) == 0:
        print("Querying PACS for study instance UIDs...")
        # Call the query function to get study_instance_uid_list
        study_instance_uid_list = query(ae, PACS_IP, PACS_PORT, CALLED_AE_TITLE, query_dict)
        sleep(10)  # Wait for 60 seconds before retrying if no UIDs found

    ae = AE(ae_title=CALLING_AE_TITLE)
    # ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
    # ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
    retrieve(ae, PACS_IP, PACS_PORT, CALLED_AE_TITLE, study_instance_uid_list, ROOT_FOLDER=output_folder)


if __name__ == "__main__":
    main()
