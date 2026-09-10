# Customer Churn Prediction System

An end-to-end machine learning system that predicts whether a telecom customer is likely to churn.

The project covers the complete machine learning workflow:

- Data collection
- Data cleaning
- Exploratory data analysis
- Data preprocessing
- Model training
- Model evaluation
- Model saving with Joblib
- Customer prediction
- Streamlit deployment
- FastAPI deployment
- API testing

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- FastAPI
- Uvicorn
- Pydantic
- pytest
# Customer Churn Prediction System

## Project overview
A beginner-friendly machine learning system that predicts whether a Telco customer is likely to churn. It demonstrates the complete pipeline: problem definition, data collection, cleaning, preprocessing, model training, evaluation, model saving/loading, Streamlit deployment, FastAPI deployment, and API testing.

## Problem definition
Customer churn is a binary classification problem:
- `0` = customer stays
- `1` = customer churns

The system helps a telecom business identify customers who may leave so the business can investigate retention actions. The model predicts the probability that a customer will churn.

## Dataset
This project uses the IBM Telco Customer Churn CSV dataset. The project does not generate fake customer records.

Place the downloaded CSV here:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

If your downloaded file has a different filename, rename it to the filename above or update the path in `src/data_collection.py`.

## Features
The model uses the dataset's customer attributes after dropping `customerID`, including demographic, service, contract, billing, tenure, and charge fields. `Churn` is the target.

## Preprocessing
- Remove duplicate rows.
- Convert `TotalCharges` to numeric; invalid/blank values become missing values.
- Convert `Churn` from `Yes/No` to `1/0`.
- Drop `customerID` because it is an identifier, not a useful predictive feature.
- Numerical columns: median imputation + `StandardScaler`.
- Categorical columns: most-frequent imputation + `OneHotEncoder(handle_unknown='ignore')`.
- All preprocessing is inside the same scikit-learn pipeline as the model, so training and prediction use identical transformations.

## Models
The training script compares:
- Logistic Regression
- Decision Tree
- Random Forest

Models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix. The selected model is the one with the best F1-score, using ROC-AUC as the tie-breaker.

## Model saving
The complete preprocessing + model pipeline is saved with Joblib as:

`models/churn_model.joblib`

## Installation - Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the project with `.venv\Scripts\python.exe` without activating the environment.

## Data collection/check

```powershell
python src/data_collection.py
```

## Train the model

First put the IBM Telco CSV in `data/raw/` as described above, then:

```powershell
python src/train.py
```

This creates `models/churn_model.joblib` and `models/evaluation_results.json`.

## EDA notebook

Open `notebooks/01_eda.ipynb` in Jupyter or VS Code after placing the dataset in `data/raw/`.

## Run Streamlit

```powershell
streamlit run app/streamlit_app.py
```

## Run FastAPI

```powershell
uvicorn api.main:app --reload
```

Swagger UI will be available at `/docs` on the local API server.

## Test the API

```powershell
pytest
```

The API tests cover `/`, `/health`, `/predict`, and invalid input. The prediction test can return `503` before the model is trained; after training, it validates the prediction response itself.

## Example API request

POST `/predict` with JSON:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0
}
```

Example response after training:

```json
{
  "prediction": 1,
  "churn_probability": 0.82,
  "risk_level": "High"
}
```

The probability above is only an example of the response format; the real value comes from the trained model.
