### Telco Customer Churn Prediction & Risk Segmentation

A machine learning application designed to quantify customer retention risk and predict churn probabilities using Random Forest and Logistic Regression models.

Live Interactive App: [View Streamlit App](https://customerchurnprediction-jpxe83e4tn57g9nvkjhevx.streamlit.app/)

## Project Overview
* Data Processing & EDA: Cleaned customer demographics, contract types, and service usage features.
* Model Evaluation: Benchmarked classifiers using Precision, Recall, and ROC-AUC metrics to optimize for retention strategy.
* Interactive Dashboard: Deployed a Streamlit UI displaying real-time predictions, annual value at risk, and key feature weights.

### Key Insights & Drivers

* Primary Predictors: `TotalCharges`, `MonthlyCharges`, and `tenure` account for over 54% of model predictive power.
* High-Risk Segment: Customers on **month-to-month contracts** using **electronic check payments** exhibit the highest overall churn rate.
* Risk Tiers: Automatically categorizes customers into High (🔴), Medium (🟡), and Low (🟢) risk buckets for automated retention workflows.

### Repository Structure

* `app.py`: Streamlit web interface and layout code.
* `churn_random_forest_model.pkl`: Serialized Random Forest machine learning model.
* `churn_scaler.pkl`: Saved feature scaler.
* `requirements.txt`: Environment dependencies for production deployment.

### Local Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/Joy-Kavata/Customer_Churn_Prediction.git](https://github.com/Joy-Kavata/Customer_Churn_Prediction.git)

a. Install Dependencies
pip install -r requirements.txt

b. Run the Streamlit app
streamlit run app.py
