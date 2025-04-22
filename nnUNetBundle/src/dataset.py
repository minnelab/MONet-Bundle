
import pathlib
import os

def get_subfolder_dataset(data_dir,modality_conf):
    data_list = []
    for f in os.scandir(data_dir):

        if f.is_dir():
            subject_dict = {key:str(pathlib.Path(f.path).joinpath(f.name+modality_conf[key]['suffix'])) for key in modality_conf}
            data_list.append(subject_dict)
    return data_list
