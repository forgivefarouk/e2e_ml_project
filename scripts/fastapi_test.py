from fastapi import FastAPI , Request
from JobPrediction import JobPrediction

RUN_ID ='a6a1acf9579f4b858116fa024bff1619'
MLFLOW_TRACKING_URI = '../models/mlruns'
EXPERIMENT_ID = 1

LOG_DATA_PKL    =  "data_rf.pkl"
LOG_MODEL_PKL   =  "model_rf.pkl"
LOG_METRICS_PKL =  "metrics_rf.pkl"

CLUSTERS_PATH = "../data/processed/skills_group_clusters.pkl"

app = FastAPI()

jobmodel = JobPrediction(mlflow_uri=MLFLOW_TRACKING_URI,run_id=RUN_ID,cluster_group_path=CLUSTERS_PATH)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_jobs_probs")
async def predict_jobs_probs(request: Request):
    available_skills = await request.json()
    predictions = jobmodel.predict_jobs_probabilities(available_skills).sort_values(ascending=False).to_dict()
    return predictions

    
