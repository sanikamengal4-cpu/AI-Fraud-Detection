# AI Digital Payment Fraud Pattern Detector

An educational Flask, scikit-learn and SQLite application for exploring payment-risk patterns. It scores a submitted transaction as **Legitimate**, **Suspicious** or **Potentially Fraudulent**, with a probability-based risk score from 0 to 100.

> This project is an educational prototype. It uses synthetic data and must not be used to approve, reject or investigate real payments.

## Run on Windows in Visual Studio Code

1. Open this folder in VS Code and select the workspace `.venv` interpreter if prompted.
2. In the integrated PowerShell terminal, install the dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   .\.venv\Scripts\python.exe run.py
   ```

4. Open `http://127.0.0.1:5000`.

On first launch, the app generates a reproducible 3,200-row synthetic dataset at `data/sample_transactions.csv`, runs EDA and saves its summary and two Seaborn/Matplotlib plots under `data/`, trains and evaluates the model, saves the preprocessing pipeline at `data/fraud_pipeline.joblib`, and creates `data/fraud_detector.db`. These local artifacts persist between restarts. The first launch takes longer while the model is trained.

## Project map

- `run.py` starts the Flask development server.
- `fraud_detector/data.py` creates and cleans the deterministic synthetic dataset.
- `data/eda_summary.json`, `data/eda_class_and_amount.png` and `data/eda_type_rates.png` are generated from the cleaned dataset and displayed on the model evaluation page.
- `fraud_detector/model.py` performs a stratified train/test split, fits preprocessing only on training rows, trains a class-weighted Random Forest, computes held-out metrics and local feature sensitivity, and saves the joblib artifact.
- `fraud_detector/database.py` initializes and writes the SQLite transaction, prediction and model-evaluation tables.
- `fraud_detector/routes.py` serves the overview, analysis, history, transaction detail, analytics, model evaluation and JSON analytics endpoints.
- `templates/` and `static/` contain the responsive Bootstrap-based interface, charts and sample form helpers.

## Model and interpretation

Input features include amount, transaction type and hour, sender/receiver balances, recent transaction count, device, location and account age. Missing values are imputed; numeric values are scaled and categorical values one-hot encoded inside one scikit-learn pipeline. The Random Forest uses balanced subsample class weights. Precision, recall, F1, ROC-AUC, accuracy and the confusion matrix are computed on the held-out test partition and stored in the database with each training run.

Each transaction detail page reports one-feature-at-a-time model sensitivity: the app replaces one submitted feature at a time with a neutral training-set value and compares its fraud score. This is a local model comparison, not a causal explanation. The model page separately reports global impurity-based feature importance.

Risk bands are derived from the predicted fraud probability: **Low** is below 25, **Medium** is 25 through 64, and **High** is 65 or above. The class labels shown in the app follow those bands.

## Database

SQLite creates three tables: `transactions`, `prediction_results` (one linked result per submission) and `model_evaluation_history`. The analytics charts and history table query saved records; filtering supports transaction ID, type, risk, class and date, with ten rows per page.

## Notes

- Synthetic labels are generated from seeded patterns such as large transfers, unusual hours, high transaction velocity, young accounts and non-local locations. The learned scores reflect those synthetic rules, not real-world fraud prevalence.
- External assets (Bootstrap, Chart.js and Google Fonts) are loaded from CDNs; the backend and database run locally.
- Delete `data/fraud_pipeline.joblib` to retrain at next startup. The existing SQLite evaluation history is retained.