

10 Academy: Artificial Intelligence Mastery
Week 4 Challenge Document

Credit Risk Probability Model for Alternative Data

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model

Date: 10 Dec - 16 Dec 2025












Overview
Business Need
You are an Analytics Engineer at Bati Bank, a leading financial service provider with over 10 years of experience. Bati Bank is partnering with an upcoming successful eCommerce company to enable a buy-now-pay-later service - to provide customers with the ability to buy products by credit if they qualify for the service. You are assigned a project to create a Credit Scoring Model using the data provided by the eCommerce platform.

Credit scoring is the term used to describe the process of assigning a quantitative measure to a potential borrower as an estimate of how likely the default will happen in the future. Traditionally, creditors build credit scoring models using statistical techniques to analyze various information of previous borrowers in relation to their loan performance. Afterward, the model can be used to evaluate a potential borrower who applies for a loan by providing similar information that has been used to build the model. The result is either a score that represents the creditworthiness of an applicant or a prediction of whether an applicant will default in the future.
The definition of default in the context of credit scoring may vary between financial institutions, as long as it complies with the Basel II Capital Accord - you must read this reference to understand the factors the bank needs to take into account to start a new loan procedure. A quick summary of the Basel II Capital Accord can be found in this reference.
The key innovation lies in transforming behavioral data into a predictive risk signal. By analyzing customer Recency, Frequency, and Monetary (RFM) patterns, we engineer a proxy for credit risk. This allows us to train a model that outputs a risk probability score, a vital metric that can be used to inform loan approvals and terms.
Your job is to build a product that does the following 
Defines a proxy variable that can be used to categorize users as high risk (bad) or low risk (good)
Select observable features that are good predictors (have high correlation) of the default variable defined in 1)
Develop a model that assigns risk probability for a new customer 
Develop a model that assigns a credit score from risk probability estimates
Develop a model that predicts the optimal amount and duration of the loan
Data and Features
The data for this challenge can be found here. Or you can find it also here: Xente Challenge | Kaggle

Data fields
TransactionId: Unique transaction identifier on the platform
BatchId: Unique number assigned to a batch of transactions for processing
AccountId: Unique number identifying the customer on the platform
SubscriptionId: Unique number identifying the customer subscription
CustomerId: Unique identifier attached to Account
CurrencyCode: Country currency
CountryCode: Numerical geographical code of the country
ProviderId: Source provider of the Item bought.
ProductId: Item name being bought.
ProductCategory: ProductIds are organized into these broader product categories.
ChannelId: Identifies if the customer used web, Android, IOS, pay later, or checkout.
Amount: Value of the transaction. Positive for debits from the customer account and negative for credits into the customer account.
Value: Absolute value of the amount
TransactionStartTime: Transaction start time
PricingStrategy: Category of Xente's pricing structure for merchants
FraudResult: Fraud status of transaction 1 = yes, 0 = No










Learning Outcomes
Skills:
Advanced use of scikit-learn 
Feature Engineering
ML Model building and fine-tuning
CI/CD deployment of ML models  
Python logging
Unit testing  
Model management
MLOps with MLFlow
Knowledge:
Reasoning with business context
Data exploration
Predictive analysis
Machine learning 
Hyperparameter tuning
Model comparison & selection
Communication:
Reporting on statistically complex issues
Team
Kerod
Mahbubah
Filimon
Key Dates
Challenge Introduction - 10:30 AM UTC on Wednesday, 10 Dec 2025.
Interim Submission - 8:00 PM UTC on Sunday, 14 Dec 2025. 
Final Submission - 8:00 PM UTC on Tuesday, 16 Dec  2025.
Communication & Support
Slack channel: #all-week-4
Office hours: Mon–Fri, 08:00–15:00 UTC
Deliverables and Tasks to be done
Project Structure
Mandate a standardized project structure from the beginning. This is a core engineering discipline.
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
Task 1 - Understanding Credit Risk
Objective:  Understand the fundamentals of credit risk and how it applies to this project.
Focus on understanding the concept of Credit Risk. Read the provided references on Credit Risk and the Basel II Capital Accord.

Instructions:
Read the following key references
https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf
https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf
https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf
https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03
https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/
https://www.risk-officer.com/Credit_Risk.htm
In your README.md file, create a section titled "Credit Scoring Business Understanding." Write a concise summary that answers the following questions:
How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Deliverables:
The updated README.md file in your GitHub repository containing the completed "Credit Scoring Business Understanding" section.
Task 2 - Exploratory Data Analysis (EDA)
Objective: Explore the dataset to uncover patterns, identify data quality issues, and form hypotheses that will guide your feature engineering.

Use Jupyter Notebook (notebooks/eda.ipynb) for all exploratory work. This notebook is for exploration only; it is not for production code.
Instructions:
Overview of the Data:
Understand the structure of the dataset, including the number of rows, columns, and data types.
Summary Statistics
Understand the central tendency, dispersion, and shape of the dataset’s distribution.
Distribution of Numerical Features
Visualize the distribution of numerical features to identify patterns, skewness, and potential outliers.
Distribution of Categorical Features
Analyzing the distribution of categorical features provides insights into the frequency and variability of categories.
Correlation Analysis
Understanding the relationship between numerical features.
Identifying Missing Values
Identify missing values to determine missing data and decide on appropriate imputation strategies.
Outlier Detection
Use box plots to identify outliers.
Deliverables:
Based on your findings, summarize your top 3–5 most important insights in your notebook.
Task 3 - Feature Engineering
Objective: Build a robust, automated, and reproducible data processing script that transforms raw data into a model-ready format.
Use sklearn.pipeline.Pipeline to chain together all transformation steps.
Instructions:
Create Aggregate Features
	Example:
Total Transaction Amount: Sum of all transaction amounts per customer.
Average Transaction Amount: Average transaction amount per customer.
Transaction Count: Number of transactions per customer.
Standard Deviation of Transaction Amounts: Variability of transaction amounts per customer.
Extract Features
	Example:
Transaction Hour: The hour of the day when the transaction occurred.
Transaction Day: The day of the month when the transaction occurred.
Transaction Month: The month when the transaction occurred.
Transaction Year: The year when the transaction occurred.
Encode Categorical Variables
Convert categorical variables into numerical format by using:
One-Hot Encoding: Converts categorical values into binary vectors. or
Label Encoding: Assigns a unique integer to each category.
Handle Missing Values
	Use imputation or Removal to handle missing values
Imputation: Filling missing values with the mean, median, mode, or using KNN imputation.
Removal: Removing rows or columns with missing values if appropriate.
Normalize/Standardize Numerical Features
Normalization and standardization are scaling techniques used to bring all numerical features onto a similar scale.
Normalization: Scales the data to a range of [0, 1].
Standardization: Scales the data to have a mean of 0 and a standard deviation of 1.
Feature Engineering with WoE and IV
Read about Weight of Evidence and Information Value
Implement WoE transformation using libraries like xverse or woe
Task 4 - Proxy Target Variable Engineering 
Objective: Create a credit risk target variable since you do not have a pre-existing "credit risk" column in your data.

You will programmatically identify "disengaged" customers and label them as high-risk proxies. High-risk groups are those with a high likelihood of default, meaning they may not pay the loan principal and interest within the specified timeframe.
Instructions:
Calculate RFM Metrics
For each CustomerId, calculate their Recency, Frequency, and Monetary (RFM) values from the transaction history.
Define a snapshot date to calculate Recency consistently.
Cluster Customers
Use the K-Means clustering algorithm to segment customers into 3 distinct groups based on their RFM profiles.
Pre-process (e.g., scale) the RFM features appropriately before clustering to ensure that the results are meaningful.
Set a random_state during clustering to ensure reproducibility.
Define and Assign the "High-Risk" Label
Analyze the resulting clusters to determine which one represents the least engaged (highest-risk) customer segment (typically characterized by low frequency and low monetary value).
Create a new binary target column named is_high_risk.
Assign a value of 1 to customers in the high-risk cluster and 0 to all others.
Integrate the Target Variable
Merge this new is_high_risk column back into your main processed dataset for model training.
Task 5 - Model Training and Tracking
Objective: Develop a structured model training process that includes experiment tracking, model versioning, and unit testing.
Instructions:
Setup
Add mlflow and pytest to your requirements.txt.
Data Preparation
Split the data into training and testing sets; this helps evaluate the model’s performance on unseen data.
Ensure reproducibility by setting a random_state.
Model Selection and Training
Choose and train at least two models from the following:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting (e.g., XGBoost, LightGBM)
Hyperparameter Tuning
Improve model performance using hyperparameter tuning techniques:
Grid Search
Random Search 
Experiment Tracking with MLflow
Log all experiments to MLflow, including:
Model parameters
Evaluation metrics
Model artifacts
Compare model runs in the MLflow UI.
After experimenting, identify your best model and register it in the MLflow Model Registry.
Model Evaluation
Assess model performance using the following metrics:
Accuracy: Ratio of correctly predicted observations to total observations
Precision: Ratio of correctly predicted positives to total predicted positives
Recall (Sensitivity): Ratio of correctly predicted positives to all actual positives
F1 Score: Weighted average of Precision and Recall
ROC-AUC: Area Under the ROC Curve
Write Unit Tests
In tests/test_data_processing.py, write at least two unit tests for helper functions within your scripts.
Example: Test that your feature engineering function returns the expected columns.
Task 6 - Model Deployment and Continuous Integration
Objective: Package the trained model into a containerized API and set up a CI/CD pipeline to automate testing and ensure code quality.
Instructions:
Setup
Add fastapi, uvicorn, and a linter (flake8 or black) to your requirements.txt.
Create the API
In src/api/main.py, build a REST API using FastAPI.
The API should load your best model from the MLflow registry.
Create a /predict endpoint that accepts new customer data (matching the model's features) and returns the risk probability.
Use Pydantic models in src/api/pydantic_models.py for request and response data validation.
Containerize the Service:
Write a Dockerfile that sets up the environment and runs the FastAPI application using uvicorn.
Write a docker-compose.yml file to easily build and run your service.
Configure CI/CD
In .github/workflows/ci.yml, create a GitHub Actions workflow that triggers on every push to your main branch.
The workflow must include:
A step to run a code linter (e.g., flake8) to check for code style issues
A step to run pytest to execute your unit tests
The build should fail if either the linter or the tests fail.



Tutorials Schedule
Overview
In the following, the colour purple indicates morning sessions, and blue indicates afternoon sessions.
Wednesday: 
Introduction to the challenge(Kerod)
Introduction to Credit Risk Analysis and Modeling (Filimon)
Thursday: 
Feature Engineering, WEIGHT OF EVIDENCE(WoE), and INFORMATION VALUE (IV) (Semegn)
Model Training, Hyperparameter Tuning, and Evaluation (Mahbubah)
Friday:
Model Serving and Deployment with Docker (Filimon)
CI/CD + Unit Testing (Mahbubah)
Monday:
Q&A(Mahbubah)
Tuesday:
Q&A(Filimon)








Due Date (Submission)
Interim Submission – Sunday, 14 Dec 2025, 8:00 PM UTC
What to Submit:
GitHub link to your main branch, showing merged work from task-1 (Business Understanding in README.md), and task-2 (EDA notebook with top 3–5 insights)
Interim report summarizing your project understanding and EDA findings (task-1 and task-2)
Final Submission – Tuesday, 16 Dec 2025, 8:00 PM UTC
What to Submit:
Link to your GitHub repository (main branch).
A polished final report in the format of a Medium blog post. This should be a self-contained, professional artifact that includes:
Business problem and proxy variable justification
RFM clustering methodology for defining high-risk customers
Model comparison with metrics
API demonstration with sample request/response
Limitations of the proxy-based approach
Screenshots: MLflow tracking, CI/CD status, Docker running
Feedback
You may not receive detailed comments on your interim submission, but you will receive a grade.






References
Credit Risk Fundamentals
https://www.investopedia.com/terms/c/creditrisk.asp
https://investopedia.com/terms/c/creditspread.asp
https://cleartax.in/glossary/credit-risk/
https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/
https://www.risk-officer.com/Credit_Risk.htm
https://drive.google.com/drive/folders/1pAXmJ_SI46D4Ex-nV0pDGvpxa7HD5erW?usp=drive_link
https://shichen.name/scorecard/
Publications:
Credit Risk Determinants in Selected Ethiopian Commercial Banks: A Panel Data Analysis
Factors Affecting Credit Risk Exposure of Commercial Banks in Ethiopia: An Empirical Analysis
Credit Risk Analysis of Ethiopian Banks: A Fixed Effect Panel Data Model. 
Feature Engineering (WoE/IV)
Weight of Evidence and Information Value Explained
xverse – Python Package
woe – Python Package
WoE Credit Scoring – GitHub
Scorecard Package Documentation
Machine Learning & MLOps
Loss Functions for Machine Learning
Sklearn Pipelines Guide
Merging DataFrames in Pandas
Random Forests Explained
Hyperparameter Tuning with GridSearchCV
Random Search vs Grid Search
Auto-sklearn Documentation
Hyperparameter tuning. Grid search and random search
Dataset
https://www.kaggle.com/datasets/atwine/xente-challenge
Additional Reading
Alternative Credit Scoring – HKMA
Credit Scoring Approaches Guidelines – World Bank
Credit Scoring Statistical Analysis
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8860138/
Credit Risk Determinants in Selected Ethiopian Commercial Banks: A Panel Data Analysis
Factors Affecting Credit Risk Exposure of Commercial Banks in Ethiopia: An Empirical Analysis
Credit Risk Analysis of Ethiopian Banks: A Fixed Effect Panel Data Model. 
