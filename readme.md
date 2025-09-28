# Purchase Prediction ML Pipeline with Azure ML Studio

## Overview
This repository showcases a **machine learning pipeline** built in **Azure ML Studio** for a **purchase prediction** use case. The pipeline demonstrates how to design, train, and deploy ML models using Azure ML’s managed services.

### Author: SAIDA.D

## Key Features
- **Azure ML Studio** pipeline design
- **Compute Cluster (AmlCompute)** for model training
- **Azure Kubernetes Service (AKS)** cluster for real-time inference
- End-to-end **MLOps workflow** (data ingestion → training → deployment)

![Azure AI ML](assets/job_done_training.png)

## Architecture
1. **Data Preparation**  
   Data assets registered in Azure ML from CSV files.

2. **Training**  
   - Training script executed on a **Compute Cluster**  
   - Model evaluation and metrics logging with MLflow

3. **Model Registration**  
   Trained models registered in the Azure ML Model Registry.

4. **Deployment**  
   - Model deployed to an **AKS cluster** for real-time inference  
   - REST endpoint exposed for predictions

## AKS Cluster
![Azure AI ML](assets/AKS_cluster.png)

## Realtime Inference
![Azure AI ML](assets/realtime_inference.png)

## Evaluation
![Azure AI ML](assets/evaluation.png)


## Getting Started
### Prerequisites
- Azure subscription
- Azure Machine Learning Workspace
- Python SDK (`azureml-sdk`)

### Clone this repository
```bash
git clone https://github.com/your-username/azureml-purchase-pipeline.git
cd azureml-purchase-pipeline
```

### Run Training
Submit training to a compute cluster:
```python
from azureml.core import Experiment, ScriptRunConfig

src = ScriptRunConfig(source_directory=".", script="train.py", compute_target="cpu-cluster")
exp = Experiment(workspace=ws, name="purchase-prediction")
run = exp.submit(src)
run.wait_for_completion(show_output=True)
```

### Deploy Model
```python
from azureml.core.webservice import AksWebservice, Webservice

aks_config = AksWebservice.deploy_configuration(cpu_cores=2, memory_gb=4)
service = Model.deploy(ws, "purchase-prediction-api", [model], inference_config, aks_target, aks_config)
service.wait_for_deployment(show_output=True)
```

## Use Case
- Predicting whether a user will make a purchase based on historical features.
- Business value: better targeting, improved marketing ROI, and personalized recommendations.

## 🔒 Note
While I have worked with Azure ML extensively in production, I have not published previous projects on GitHub due to confidentiality. This repository is **for demonstration and showcase purposes only**.

---



