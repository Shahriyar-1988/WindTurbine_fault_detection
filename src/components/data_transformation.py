from src.entity.config_entity import DataTransformationConfig
from src import logger
from src.constants import SCHEMA_FILE_PATH
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from src.utils.common import read_yaml,save_bin
import pandas as pd
import os

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
    @staticmethod    
    def load_schema():
        schema=read_yaml(SCHEMA_FILE_PATH)
        drop_col=schema["DORP_COLS"]
        target_col=schema["TARGET_COLUMN"]
        normal_label=schema["NORMAL_LABEL"]
        idling_label=schema["IDLING_LABEL"]
        return drop_col, target_col,normal_label,idling_label
    def data_transformer(self):
        df=pd.read_csv(self.config.data_dir)
        drop_col,target_col,normal_label, idling_label=self.load_schema()
        df=df[df[target_col]!=idling_label] # removing idle as neither faulty nor normal operation
        df=df.drop(drop_col,axis=1)
        logger.info(f"Dropped irrelevant columns: {drop_col}")
        df = df.loc[:, ~(df == 0).all()]
        # drop all-zero columns[inactive sensors]
        logger.info(f"Removed columns with all zero values. Remaining columns: {df.shape[1]}")
        logger.info(f"The size of complete dataset: {df.shape}")
        # only a subset of this huge dataset is more than enough for training
        logger.info("A subset of the dataset is selected")
        splitter = StratifiedShuffleSplit(n_splits=1,train_size=0.3,random_state=32)
        for idx,_ in splitter.split(df,df[target_col]):
            df_subset=df.iloc[idx]
        logger.info(f"The size of the subset dataset: {df_subset.shape}")
        train_data,test_data=train_test_split(df_subset,test_size=0.15,random_state=42)

        # It is time to single out "normal operation only" data for training
        train_df=train_data[train_data[target_col].isin([normal_label])].copy()
        logger.info(f"Training on samples of normal operation only!")
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_data.shape}")
        X_train=train_df.drop([target_col],axis=1)

        # Defining imputer to handle missing data
        imputer=KNNImputer(n_neighbors=3)
        X_train_imputed=imputer.fit_transform(X_train)
        
        # Defining/training the scaler
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train_imputed)


        # Saving all artifacts
        scaler_path=os.path.join(self.config.root_dir,"AE_scaler.pkl")
        imputer_path=os.path.join(self.config.root_dir,"AE_imputer.pkl")
        save_bin(scaler,file_path=scaler_path), save_bin(imputer,file_path=imputer_path)
        pd.DataFrame(X_train_scaled,columns=X_train.columns).to_csv(self.config.root_dir +"/train.csv",index=False)
        test_data.to_csv(self.config.root_dir+"/test.csv",index=False)
        logger.info(f"Saved train/test data,scaler and imputer.")







