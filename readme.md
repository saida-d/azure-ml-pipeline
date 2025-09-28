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

# Traditional Approach with sklearn (Mlflow, DVC and dagshub)
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
import mlflow
import dagshub
import re
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
import json

dataset= pd.read_csv(r"data/dataset.csv")

# change tracking to remote
dagshub.init(repo_owner='YOUR_USERNAME', repo_name='purchase_predictions', mlflow=True)

dags_token="XXXXXXXXXXX66508522049056df62a01b757f7"
mlflow.set_experiment("Purchase Prediction - Multi Model")
mlflow.set_tracking_uri(f"https://{dags_username}:{dags_token}@dagshub.com/YOUR_USERNAME/purchase_predictions.mlflow")
 

class DataPreprocessor:

	# a class variable
    learning_rate=0.1

    def __init__(self, dataset):
        self.dataset=dataset.copy() # ignore last column
        self.target=dataset.iloc[:,-1] # only last col
        self.X_train=''
        self.X_test=''
        self.y_train=''
        self.y_test=''

    def data_preprocess(self):
        # check duplicates
        if(self.dataset.duplicated().sum()>0):
            # remove duplicates
            self.dataset.drop_duplicates()
           
        # check null values
        null_count=self.dataset.isnull().sum()

        # impute null values  
        lst_impute=['Feature_1', 'Feature_2']
        for idx,null_total in null_count.items():
            if(null_total>0):
                if f'{idx}' in lst_impute:
                    idx_mean= self.dataset[idx].mean()
                    self.dataset[idx].fillna(idx_mean, axis=0, inplace=True)
                else:
                    self.dataset = self.dataset.dropna(subset=[idx])

        if 'Country' in self.dataset.columns:
            self.dataset['Country'] = self.dataset['Country'].replace(
                {"]Italy&": "Italy", "]Germany&": "Germany", "Germany&": "Germany"})          
        
        if 'ID' in self.dataset.columns:
            # drop user Id it wont impact
            self.dataset.drop('ID', axis=1, inplace=True)

        if 'Date' in self.dataset.columns:
            self.dataset['Date']   = pd.to_datetime(self.dataset['Date'], errors='coerce')
            self.dataset           = self.dataset.dropna(subset=['Date'])
            self.target            = self.target.loc[self.dataset.index]
            self.dataset['Year']   = self.dataset['Date'].dt.year
            self.dataset['Month']  = self.dataset['Date'].dt.month
            self.dataset['Day']    = self.dataset['Date'].dt.day
            self.dataset           = self.dataset.drop('Date', axis=1)
        
        if 'EstimatedSalary' in self.dataset.columns:
            Q1           = self.dataset["EstimatedSalary"].quantile(q = 0.25)
            Q3           = self.dataset["EstimatedSalary"].quantile(q = 0.75)
			IQR          = Q3-Q1
		    lower_bound  = Q1- (1.5 * IQR)
            upper_bound  = Q3+ (1.5 * IQR)
            outliers     = self.dataset[(self.dataset.EstimatedSalary < lower_bound) | (self.dataset.EstimatedSalary>upper_bound)]
        
            if(len(outliers)>0):
                self.dataset.drop(outliers.index, axis=0, inplace=True)

        return self.dataset      
    
   
    def encoded_data(self):
        dataset=self.data_preprocess()
        oh       = OneHotEncoder(sparse_output=False)
        ct       = ColumnTransformer(transformers=[('ohe', oh, ['Gender', 'Country'])], 
                              remainder='passthrough')
        dataset  = ct.fit_transform(dataset)
        column_names = []
		
        for each_col in ct.get_feature_names_out():
            column_names.append(each_col.split("_")[-1])
        self.dataset = pd.DataFrame(dataset, columns=column_names)
        
        return self.dataset
    
    def train_test_data_split(self):
        self.encoded_data()
        # train test split
        X      = self.dataset[["Feature_1", "Feature_2"]]
        y      = self.dataset["Target"]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)
        
        self.X_train  = X_train
        self.X_test   = X_test
        self.y_train  = y_train
        self.y_test   = y_test
        print('before smote {}'.format(self.y_train.value_counts()))
        
        return 'train_test_split'
   

    def smote(self):
        # apply SMOTETomek (or) SOMTE from Pipeline to balance training data
        smote_tomek= SMOTETomek(random_state=45)
        X_train_resampled, y_train_resampled=smote_tomek.fit_resample(self.X_train,self.y_train)

        print('after smote {}'.format(y_train_resampled.value_counts()))
        return X_train_resampled, y_train_resampled
    
    def split_with_scale(self):
        self.train_test_data_split()
        X_train_resampled, y_train_resampled=self.smote()

        # scaling
        sc           = StandardScaler()
        self.X_train = sc.fit_transform(X_train_resampled)
        self.X_test  = sc.transform(self.X_test)
        self.y_train = y_train_resampled
        
        return 'split_with_smote_scale'
    
    def split_with_pca(self):
        self.train_test_data_split()
        pca_data      = PCA(n_components=2)
        self.X_train  = pca_data.fit_transform(self.X_train) 
        self.X_test   = pca_data.transform(self.X_test)
        return 'split_with_pca'
    
 
class PurchasePredict(DataPreprocessor):
    def __init__(self, dataset):
        super().__init__(dataset)
        
    def logistic_regression(self):
        
        # data with scale
        self.split_with_scale()
        
        # logistic regression classifier
        lor_params   = {'max_iter':500,'penalty':'l2','C':0.25}
        lor_model    = LogisticRegression(
			max_iter = lor_params['max_iter'], 
            penalty  = lor_params['penalty'],
            C        = lor_params['C']
           
        )

        # train model
        lor_model.fit(self.X_train,self.y_train)
       
        model= {'model':lor_model,'name':'LogisticRegression','params':lor_params}
        return self.train_metrics(model)
    
    
    def svm(self):

        # data with scale
        self.split_with_scale()
        
        # SVM Classifier
        svm_params={
            'kernal':'rbf',
            'gamma':10, 
            'C':0.2, 
            'max_iter':100
        }
        svm_model=SVC(kernel='rbf',gamma=10,C=0.2,max_iter=100)

        # train model
        svm_model.fit(self.X_train,self.y_train)
        model={'model':svm_model,'name':'SVM-Classifier(SVC)','params':svm_params}

        return self.train_metrics(model)

    def knn(self):

        # data no scale
        self.train_test_data_split()
        
        # KNN classifier
        knn_params={'n_neighbors':5}
        knn_model=KNeighborsClassifier(n_neighbors=knn_params['n_neighbors'])

        # train model
        knn_model.fit(self.X_train,self.y_train)
        model={'model':knn_model,'name':'KNN-Classifier','params':knn_params}

        return self.train_metrics(model)
    
    def decision_tree(self):

        # data no scale
        self.train_test_data_split()
        
        # Decision tree
        dt_params={
            'max_depth':5,
            'min_samples_split':35,
            'criterion':'gini'
        }
        dt_model=DecisionTreeClassifier(
            criterion         = dt_params['criterion'],
            max_depth         = dt_params['max_depth'],
            min_samples_split = dt_params['min_samples_split']
        )
        
        # train model
        dt_model.fit(self.X_train,self.y_train)
        
        model={'model':dt_model,'name':'Decision Tree-Classifier','params':dt_params}
        return self.train_metrics(model)
    
    def random_forest(self):
        # data no scale
        self.train_test_data_split()
        
        rf_params={'n_estimators':100, 'criterion':"gini" , 'max_depth':3}
        rf_model=RandomForestClassifier(
            n_estimators  = rf_params['n_estimators'], 
            criterion     = rf_params['criterion'], 
            max_depth     = rf_params['max_depth']
        )
        rf_model.fit(self.X_train,self.y_train)
        model={'model':rf_model,'name':'Random Forest Tree-Classifier','params':rf_params}
		
        return self.train_metrics(model)
    
    def train_metrics(self,model):
	
		# predict over train/test data 
        y_train_pred=model['model'].predict(self.X_train)
        y_test_pred=model['model'].predict(self.X_test)
        
        # log metrics to mlflow
        with mlflow.start_run(run_name=model['name']) as run:
            mlflow.log_params(model['params'])
            mlflow.log_metric('train_accuracy_score',accuracy_score(self.y_train,y_train_pred))
            mlflow.log_metric('train_f1_score',f1_score(self.y_train,y_train_pred))
            
            mlflow.log_metric('test_accuracy_score',accuracy_score(self.y_test,y_test_pred))
            mlflow.log_metric('test_f1_score',f1_score(self.y_test,y_test_pred))
            mlflow.sklearn.log_model(model['model'],model['name'])
      
        return model
    
    def dagshub_inference(self,model):
        model_url=f"models:/{model['name']}@champion"
		
		#load best model 
        best_prod_model=mlflow.sklearn.load_model(model_url)
		
		#predict over test
        y_prod_pred=best_prod_model.predict(self.X_test)
		
        return y_prod_pred

```

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



