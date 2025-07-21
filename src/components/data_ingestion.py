import os
from src import logger
from src.config.configuration import DataIngestionConfig
from dotenv import load_dotenv
import kagglehub
import shutil
from src.constants import *
import pandas as pd

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
        load_dotenv()
    def download_data(self):
        # set Kaggle credintials
        os.environ["KAGGLE_USERNAME"]=os.getenv("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"]=os.getenv("KAGGLE_KEY")
        dataset_name=self.config.data_url
        logger.info(f"Downloading dataset: {dataset_name}")
        #Kaggle download command will download dataset to windows directory
        path=Path(kagglehub.dataset_download(dataset_name))
        # Extracting the downloaded dataset from zipfile
    
        for item in path.iterdir():
            dest = Path(self.config.data_dir) / item.name
            if item.is_file():
                shutil.copy(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)

        logger.info(f"Dataset copied to {self.config.data_dir}")
        
    def combined_data(self):
        combined_file_path = Path(self.config.data_dir) / "combined_raw_data.csv"
        try:
            if combined_file_path.exists():
                logger.info(f"Combined file already exists at: {combined_file_path}, skipping combination.")
            else:
                all_files=list(Path(self.config.data_dir +"\Wind Farm B\datasets").glob('*.csv'))
                logger.info(f"Combining {len(all_files)} CSV files.")
                df_list = [pd.read_csv(f) for f in all_files]
                combined_df = pd.concat(df_list, ignore_index=True)
                combined_df.to_csv(combined_file_path, index=False)
                logger.info(f"Combined CSV saved at: {combined_file_path}")

            return combined_file_path
        except FileNotFoundError as e:
                logger.warning("No CSV files found to combine.")

    
        
        
