# CLASSIFICATION SYSTEM FOR PREDICTING OUTCOME IN PATIENCE WITH MYOCARDIAL INFARCTION
When patients are hospitalized for myocardial infarction (MI), the first 3 days are critical because of the high chances of fatality from various types of complications. There is a state-of-the-art paper by Ghafari et al, (see paper [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC11053239/)), which built a machine learning model to predict fatality in myocardial infarction patients within their first 3 days of admission. I identified 2 major loophole in the paper which I decided to build a new model with the same data to address these loophole and also produce a full production grade machine learning system for prediction of outcome in MI patients. The paper was aimed at predicting fatality however, instead of treating the class for the patients that died within 3 days in the training set as the positive class (that is the class to be predicted), the paper treated the class of those that survived the first 3 days as positive class. With this, sensitivity and specificity were interchanged and sensitivity which is the most important matric was reported to be 0.9435 instead of 0.6923 which was wrongly reported as the specificity. The second problem identified was the common problem of class imbalance which is normal for most real world dataset. The use of methods like SMOTE (which is proven to improve performance significantly when used to address class imbalance) was ignored. In my new model, I designated the positivie and negetive classes correctly and used SMOTE to address class imbalance thereby significantly improving the performance to at least 97% for all metrices. While the best performing algorithm in the paper was XGBoost, I identified a new algorithm which performs better than XGBoost. I thereafter developed the project into a full production grade machine learning system that can be integreated into any web or mobile applicatio. The steps and components of the machine learning system are discuused below:

## PROJECT OVERVIEW/STEPS
### Dataset
The dataset used in this project was collected from 1699 myocardial infarction (MI) patients admitted to the Krasnoyarsk Interdistrict Clinical Hospital (Russia) between 1992 and 1995, aiming to predict fatal complications within the first 72 hours of hospital admission. It includes 111 clinical and demographic variables such as age, sex, obesity, MI type, ECG conduction characteristics at admission, laboratory parameters (e.g., ALT, AST, potassium, sodium levels), underlying diseases (e.g., diabetes, hypertension, chronic heart failure), signs and symptoms (e.g., blood pressure, arrhythmias), and administered drugs during emergency and ICU care. The target outcome is binary, categorizing patients into those who survived (84.04%) and those who experienced fatal complications (15.94%) caused by conditions like cardiogenic shock, pulmonary edema, myocardial rupture, and ventricular fibrillation. The dataset, with 7.6% missing values handled using mean imputation, was balanced by oversampling the minority class using SMOTE.
### Experimentation/Model Training
The model training process involved the neccessary proprocessing and experimentation, leading to the choice of the optimal model. The preprocessing steps include but not limited to diemsnionality reduction using different approach and arriving at different versions of dataset. Experimentaion involves fitting of 27 different algorithms to each of the versions of the dataset, out of which I selected the combination of the best-performing algorithm and best-performing dataset. This is how I arrived at ExtratreesClassifier applied to the balanced version of the full dataset as my optimal strategy. 
### Experiment Tracking
Experiment registry and model tracking were done using MLFlow.
### Workflow Orchestration
Workflow orchestration is necessary to enable continuous model re-training, batch inferencing and continous monitoring. All the workflows (preprocessing, model training, model evaluation, batch inferencing and continuous monitoring workfloes) were orchestrated using prefect. There are 3 different flows. The first contain all tasks from data preprocessing, model training to model evaluation. The second flow contains all tasks for batch inferencing while the third flow contains all tasks for continuous monitoring of model in production. All the workflow can either be triggered to run manually or automatically at pre-determined time interval.
### Model Serving
The model was wrapped in a FastAPI
### Remote Hosting of API
The FastAPI was deployed to AWS EC2. To access the API through the IP address, click [here](http://18.222.206.16:8000/docs#/default/predict_post_predict_post)

### Continous Integration and Continous Deployment (CI/CD)
I configured continuous integration and continuous deployment workflow for the project using GitHub Action. All the process of pusing the repository to AWS ECR and deploying the API to AWS EC2 were automated using Github Action CI/CD pipeline. 
### Offline Inferencing
Batch inferencing pipeline was also setup using Prefect. The pipeline searches through a an input folder at specified interval, picks any CSV file in the folder, preprocess the data, and use the model to assign label to the data. The confidence level for the prediction is also attached. After that, the pipeline deletes the original input from the input folder and sends the labeled dataset to an output folder.
### Model Monitoring/Full Stack Observability
Model inputs, outputs and metrics are being monitored continously for data drift, concept drift and performance decay. WhyLabs was used for this purpose.
### Other Automation
As seen earlier, all the worflows are automated meaning that there will be continuous update of the model without human intervention. Also, the CI/CD pipeline ensures automated pushing of updates to relevant AWS services (ECR and EC2). However, there is the need to also automate the pushing of updates from local repository to remote (github) repository. This was also automated with a bash script and a microsoft task that runs the script to automatically push update from local to remote repository at interval. With this, all steps have been fully automated.
### Project Versioning
Git was used for project version control
## USING THE PROJECT
The following steps can be followed to clone and run the project
 -   Clone project by running running the following command from git bash or command line:
```bash
git clone https://github.com/joagada2/mi_fatality_prediction.git
```
 -   Create a python 3.10 virtual environment and install requirements.txt including prefect 3.1.8, mlflow 2.19.0 and whylogs 1.6.4
 -   Change directory to the project by running the following command from your command line: cd mi_fatality_prediction (type cd mi and use shift + tab to autocomplete)
 -  To use the API, run the following command from your command line: python app/main.py
 -  To run the re-training pipeline, run the following command from your command line, from root directory, run cd src/prefect_orchestration and run python training_tasks.py. This will deploy the code to prefect server. Run prefect server start to start the ui and access the ui on port 4200 by clicking [here](http://localhost:4200).
 -  To run/start the batch inferencing code, from project root directory, cd to the script by running cd batch inferencing and run python batch_inf_script.py to upload the code to prefect server and run tasks every seconds
 -  To run/start the whylabs continuous monitoring script, from root directory, run cd src/whylab_monitoring and run python monitoring_script.py. This will also deploy the code to prefect server and run successfully.
 -  Note the crone configuration if for the flows to run every second. This configuration can be changed to run the code at any interval of choice
 -  Note that ETL pipeline by data engineer is expected to suppy the data for re-training, batch inferencing and monitoring. The actual crone configuration should depend on how often new data are available

 ## TOOLS/SKILLS USED IN THIS PROJECT
  - Python (Pandas, matplotlib, seaborn, scikitlearn etc)
  - Lazypredict - for experimentation
  - ExtratreesClassifiers - for model training
  - MLFlow - Experiment registry/tracking
  - FastAPI - Model Serving
  - AWS EC2 - API deployment
  - AWS ECR - Repository management on AWS
  - GitHub/GitHub Action - Code hosting and CI/CD pipeline
  - WhyLabs - Continous monitoring of model in production
  - Prefect - For workflow orchestration
  - Git - For project version control
  - etc



