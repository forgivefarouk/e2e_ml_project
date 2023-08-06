LOG_DATA_PKL = 'data_rf.pkl'
LOG_MODEL_PKL = 'model_rf.pkl'

#-------------------------------------------------------------

import os
import pickle
import pandas as pd
import numpy as np
import sklearn
import mlflow
from mlflow.tracking import MlflowClient

#-------------------------------------------------------------

class JobPrediction:
    
    def __init__(self,mlflow_uri,run_id,cluster_group_path):
        self.tracking_uri = mlflow_uri
        self.run_id = run_id
        self.cluster_group_path = cluster_group_path
        self.cluster_data = self.load_cluster_group(cluster_group_path)
        
        
        mlflow_objs = self.load_mlflow_objs()
        self.model = mlflow_objs[0]
        self.features= mlflow_objs[1]
        self.target = mlflow_objs[2]
        
        
        
    def load_mlflow_objs(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()

        run = client.get_run(self.run_id)
        artifact_path = run.info.artifact_uri
        
        with open(os.path.join(artifact_path, LOG_DATA_PKL).replace("file:///", ""), "rb") as f:
            data = pickle.load(f)
            
        with open(os.path.join(artifact_path, LOG_MODEL_PKL).replace("file:///", ""), "rb") as f:
            model= pickle.load(f)
            
            
        return model['model'] , data['feature_name'], data['tareget_name']
    
    
    
    
    def load_cluster_group(self,cluster_group_path):
        with open(self.cluster_group_path , 'rb') as f:
            cluser_data = pickle.load(f)
            
        return pd.Series(cluser_data).explode()
        
        
        
    def get_all_skills(self):
        return self.features
    
    
    
    def get_all_jobs(self):
        return self.target
    
    
    def create_features(self,skills):
        
        cluster_features = self.cluster_data.isin(skills).astype(int).groupby(level=0).sum()
        
        features_series = pd.Series(self.features)
        features_skill = features_series[~features_series.isin(self.cluster_data.index)]
        
        ohe_skils = pd.Series(features_skill.isin(skills).astype(int).to_list(), index=features_skill.values)
        
        combine_df = pd.concat([ohe_skils, cluster_features], axis=0)
        
        return combine_df.loc[self.features].values.reshape(1,-1)
    
    
    
    def predict_jobs_probabilities(self,skills):
        features = self.create_features(skills)
        pred = self.model.predict_proba(features)
        
        positive_probs = [prob[0][1] for prob in pred]
        return pd.Series(positive_probs, 
          index=self.target).sort_values(ascending=False)
