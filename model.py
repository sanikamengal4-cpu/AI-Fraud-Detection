from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .database import save_evaluation
from .data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    create_eda_artifacts,
    ensure_sample_dataset,
)

MODEL_VERSION = "rf-synthetic-v3"
DISPLAY_NAMES = {
    "amount": "Transaction amount",
    "transaction_type": "Transaction type",
    "transaction_hour": "Transaction hour",
    "sender_previous_balance": "Sender previous balance",
    "sender_new_balance": "Sender new balance",
    "receiver_previous_balance": "Receiver previous balance",
    "receiver_new_balance": "Receiver new balance",
    "transactions_last_hour": "Transactions in the last hour",
    "device_type": "Device type",
    "location": "Transaction location",
    "account_age_days": "Account age",
}


def _build_pipeline():
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ] 
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    classifier = RandomForestClassifier(
        n_estimators=160,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessing), ("classifier", classifier)])


def load_or_train_model(model_path, dataset_path, database_path):
    model_file = Path(model_path)
    regenerate_dataset = False
    if model_file.exists():
        artifact = joblib.load(model_file)
        if artifact.get("version") == MODEL_VERSION:
            eda_dir = Path(dataset_path).parent
            summary_path = eda_dir / "eda_summary.json"
            if not summary_path.exists():
                create_eda_artifacts(ensure_sample_dataset(dataset_path), eda_dir)
            return artifact
        regenerate_dataset = True

    data = ensure_sample_dataset(dataset_path, regenerate=regenerate_dataset)
    create_eda_artifacts(data, Path(dataset_path).parent)
    features = data[FEATURES]
    labels = data[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    pipeline = _build_pipeline()
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
        "fraud_rate": round(float(labels.mean()), 4),
    }
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
    artifact = {"pipeline": pipeline, "metrics": metrics, "confusion_matrix": matrix, "version": MODEL_VERSION}
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_file)
    save_evaluation(database_path, metrics, matrix)
    return artifact


def _risk_band(score):
    if score >= 65:
        return "High", "Potentially Fraudulent"
    if score >= 25:
        return "Medium", "Suspicious"
    return "Low", "Legitimate"


def predict_transaction(artifact, values):
    pipeline = artifact["pipeline"]
    record = pd.DataFrame([{name: values[name] for name in FEATURES}])
    risk = float(pipeline.predict_proba(record)[0, 1])
    score = round(risk * 100, 1)
    risk_level, predicted_class = _risk_band(score)

    # One-feature-at-a-time neutralization provides a transparent local sensitivity view.
    preprocessor = pipeline.named_steps["preprocessor"]
    medians = {}
    modes = {}
    for feature in NUMERIC_FEATURES:
        medians[feature] = float(preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_[NUMERIC_FEATURES.index(feature)])
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    for index, feature in enumerate(CATEGORICAL_FEATURES):
        modes[feature] = str(encoder.categories_[index][0])

    sensitivities = []
    for feature in FEATURES:
        neutral = record.copy()
        neutral.loc[0, feature] = medians[feature] if feature in medians else modes[feature]
        neutral_risk = float(pipeline.predict_proba(neutral)[0, 1])
        delta = (risk - neutral_risk) * 100
        if abs(delta) >= 0.5:
            sensitivities.append(
                {
                    "name": DISPLAY_NAMES[feature],
                    "impact": round(delta, 1),
                    "direction": "increases" if delta > 0 else "reduces",
                    "value": values[feature],
                }
            )
    sensitivities.sort(key=lambda item: abs(item["impact"]), reverse=True)

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "predicted_class": predicted_class,
        "factors": sensitivities[:4],
        "model_version": artifact["version"],
    }