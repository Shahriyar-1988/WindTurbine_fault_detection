import os
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError 
from pathlib import Path
import joblib
from ensure import ensure_annotations
from src import logger
from typing import Any

@ensure_annotations
def read_yaml(yaml_path:Path)->ConfigBox:
    try:
        with open(yaml_path) as f:
            content=yaml.safe_load(f)
            logger.info(f"yaml file {yaml_path} loaded successfully!")
            return ConfigBox(content)
    except Exception as e:
        raise e
    except BoxValueError:
        raise ValueError(f"yaml file {os.path.split(yaml_path)[1]} is empty!")
def create_directory(dir_path:list,verbose=True):
    try:
        for path in dir_path:
            os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at :{path}")
    except Exception:
        raise Exception
def save_bin(data:Any,file_path:Path):
    joblib.dump(data,file_path)
    logger.info(f"binary file saved at: {file_path}")
def load_bin(file_path:Path):
    loaded_data=joblib.load(file_path)
    logger.info(f"binary file loaded from: {file_path}")
    return loaded_data

    