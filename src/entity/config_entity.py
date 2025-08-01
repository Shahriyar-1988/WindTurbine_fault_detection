from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    data_url: str
    data_dir: Path
@dataclass
class DataValidationConfig:
    data_dir: Path
    status_file : str
@dataclass
class DataTransformationConfig:
    data_dir: Path
    root_dir: Path
@dataclass
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    metrics_dir: Path
    model_name: str
@dataclass
class ClassifierTrainingConfig:
    root_dir: Path
    data_path: Path
    metrics_dir: Path
    encoder_path: Path
    model_name: str
@dataclass
class ClassifierEvaluationConfig:
    test_data_paths: list
    cls_model_path: Path
    metrics_file_path: Path




  


    

    