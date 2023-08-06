MLFLOW_TRACKING = '../models/mlruns'
MLFLOW_RUN_ID = 'a6a1acf9579f4b858116fa024bff1619'
CLUSTERS_PATH = "../data/processed/skills_group_clusters.pkl"



#------------------------------------------

from JobPrediction import JobPrediction
from flask import Flask, request, jsonify

#------------------------------------------

# Initiate API and JobPrediction object
app = Flask(__name__)
job_model = JobPrediction(mlflow_uri=MLFLOW_TRACKING,
                          run_id=MLFLOW_RUN_ID,
                          cluster_group_path=CLUSTERS_PATH)


# Create prediction endpoint 
@app.route('/predict_jobs_probs', methods=['POST'])
def predict_jobs_probs():
    available_skills = request.get_json()
    predictions = job_model.predict_jobs_probabilities(available_skills).to_dict()
    return jsonify(predictions)



if __name__ == '__main__':
    app.run()
    