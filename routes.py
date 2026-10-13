import json
import sqlite3
import uuid
from contextlib import closing
from datetime import date

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

from .database import connect_database, save_prediction, transaction_detail
from .data import CATEGORICAL_FEATURES, FEATURES
from .model import predict_transaction

main = Blueprint("main", __name__)
TRANSACTION_TYPES = ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "DEBIT"]
DEVICES = ["Mobile", "Desktop", "Tablet", "Unknown"]
LOCATIONS = ["Local", "Domestic", "International", "VPN / Proxy"]
FORM_OPTIONS = {"transaction_type": TRANSACTION_TYPES, "device_type": DEVICES, "location": LOCATIONS}
NUMERIC_INPUTS = [
    "amount",
    "transaction_hour",
    "sender_previous_balance",
    "sender_new_balance",
    "receiver_previous_balance",
    "receiver_new_balance",
    "transactions_last_hour",
    "account_age_days",
]


def _database():
    return connect_database(current_app.config["DATABASE"])


def _dashboard_data():
    with closing(_database()) as connection:
        totals = connection.execute(
            """SELECT COUNT(*) AS total,
                      SUM(CASE WHEN p.predicted_class = 'Suspicious' THEN 1 ELSE 0 END) AS suspicious,
                      SUM(CASE WHEN p.predicted_class = 'Potentially Fraudulent' THEN 1 ELSE 0 END) AS fraud,
                      SUM(CASE WHEN p.risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk
               FROM transactions t JOIN prediction_results p ON p.transaction_fk = t.id"""
        ).fetchone()
        recent = connection.execute(
            """SELECT t.transaction_id, t.transaction_type, t.amount, p.predicted_class,
                      p.risk_score, p.risk_level, p.created_at, t.id
               FROM transactions t JOIN prediction_results p ON p.transaction_fk = t.id
               ORDER BY p.created_at DESC, p.id DESC LIMIT 6"""
        ).fetchall()
        split = connection.execute(
            """SELECT p.predicted_class, COUNT(*) AS count FROM prediction_results p
               GROUP BY p.predicted_class"""
        ).fetchall()
        types = connection.execute(
            """SELECT t.transaction_type,
                      SUM(CASE WHEN p.predicted_class != 'Legitimate' THEN 1 ELSE 0 END) AS flagged,
                      COUNT(*) AS total
               FROM transactions t JOIN prediction_results p ON p.transaction_fk = t.id
               GROUP BY t.transaction_type ORDER BY total DESC"""
        ).fetchall()
    total = int(totals["total"] or 0)
    return {
        "total": total,
        "suspicious": int(totals["suspicious"] or 0),
        "fraud": int(totals["fraud"] or 0),
        "detection_rate": round((int(totals["high_risk"] or 0) / total) * 100, 1) if total else 0,
        "recent": recent,
        "split": {row["predicted_class"]: row["count"] for row in split},
        "types": [dict(row) for row in types],
    }


@main.app_context_processor
def inject_navigation_context():
    return {"today": date.today().isoformat()}


@main.route("/")
def home():
    artifact = current_app.extensions["fraud_model"]
    return render_template(
        "index.html",
        stats=_dashboard_data(),
        metrics=artifact["metrics"],
        confusion=artifact["confusion_matrix"],
    )


@main.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        try:
            transaction = {name: request.form[name].strip() for name in request.form}
            if not transaction.get("transaction_id"):
                transaction["transaction_id"] = f"TX-{uuid.uuid4().hex[:10].upper()}"
            transaction["amount"] = float(transaction["amount"])
            transaction["transaction_hour"] = int(transaction["transaction_hour"])
            transaction["sender_previous_balance"] = float(transaction["sender_previous_balance"])
            transaction["sender_new_balance"] = float(transaction["sender_new_balance"])
            transaction["receiver_previous_balance"] = float(transaction["receiver_previous_balance"])
            transaction["receiver_new_balance"] = float(transaction["receiver_new_balance"])
            transaction["transactions_last_hour"] = int(transaction["transactions_last_hour"])
            transaction["account_age_days"] = int(transaction["account_age_days"])
            if transaction["amount"] <= 0 or any(
                transaction[key] < 0
                for key in [
                    "sender_previous_balance", "sender_new_balance", "receiver_previous_balance",
                    "receiver_new_balance", "transactions_last_hour", "account_age_days",
                ]
            ):
                raise ValueError("Amounts, balances, activity counts, and account age must be non-negative; amount must be greater than zero.")
            if not 0 <= transaction["transaction_hour"] <= 23:
                raise ValueError("Transaction hour must be between 0 and 23.")
            for field, choices in FORM_OPTIONS.items():
                if transaction[field] not in choices:
                    raise ValueError(f"Choose a valid {field.replace('_', ' ')}.")

            result = predict_transaction(current_app.extensions["fraud_model"], transaction)
            transaction_pk = save_prediction(current_app.config["DATABASE"], transaction, result)
            flash("Prediction saved to transaction history.", "success")
            return redirect(url_for("main.transaction_page", transaction_pk=transaction_pk))
        except (KeyError, TypeError, ValueError) as error:
            flash(str(error) or "Enter valid values for every field.", "danger")
        except sqlite3.IntegrityError:
            flash("That transaction ID already exists. Use a unique ID to keep the audit trail clear.", "danger")

    return render_template("analyze.html", options=FORM_OPTIONS)


@main.route("/transactions/<int:transaction_pk>")
def transaction_page(transaction_pk):
    row = transaction_detail(current_app.config["DATABASE"], transaction_pk)
    if row is None:
        abort(404)
    transaction = dict(row)
    transaction["factors"] = json.loads(transaction.pop("factors_json"))
    return render_template("transaction_detail.html", transaction=transaction)


@main.route("/history")
def history():
    filters = {
        "q": request.args.get("q", "").strip(),
        "transaction_type": request.args.get("transaction_type", ""),
        "risk_level": request.args.get("risk_level", ""),
        "prediction": request.args.get("prediction", ""),
        "date_from": request.args.get("date_from", ""),
        "date_to": request.args.get("date_to", ""),
    }
    clauses, values = [], []
    if filters["q"]:
        clauses.append("t.transaction_id LIKE ?")
        values.append(f"%{filters['q']}%")
    if filters["transaction_type"] in TRANSACTION_TYPES:
        clauses.append("t.transaction_type = ?")
        values.append(filters["transaction_type"])
    if filters["risk_level"] in ["Low", "Medium", "High"]:
        clauses.append("p.risk_level = ?")
        values.append(filters["risk_level"])
    if filters["prediction"] in ["Legitimate", "Suspicious", "Potentially Fraudulent"]:
        clauses.append("p.predicted_class = ?")
        values.append(filters["prediction"])
    if filters["date_from"]:
        clauses.append("date(p.created_at) >= date(?)")
        values.append(filters["date_from"])
    if filters["date_to"]:
        clauses.append("date(p.created_at) <= date(?)")
        values.append(filters["date_to"])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    page = max(1, request.args.get("page", 1, type=int))
    per_page = current_app.config["PER_PAGE"]
    with closing(_database()) as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM transactions t JOIN prediction_results p ON p.transaction_fk = t.id {where}",
            values,
        ).fetchone()[0]
        rows = connection.execute(
            f"""SELECT t.id, t.transaction_id, t.transaction_type, t.amount,
                       p.predicted_class, p.risk_score, p.risk_level, p.created_at
                FROM transactions t JOIN prediction_results p ON p.transaction_fk = t.id
                {where} ORDER BY p.created_at DESC, p.id DESC LIMIT ? OFFSET ?""",
            [*values, per_page, (page - 1) * per_page],
        ).fetchall()
    pages = max(1, (total + per_page - 1) // per_page)
    return render_template(
        "history.html", rows=rows, filters=filters, page=page, pages=pages, total=total, per_page=per_page,
        transaction_types=TRANSACTION_TYPES,
    )


@main.route("/analytics")
def analytics():
    artifact = current_app.extensions["fraud_model"]
    with closing(_database()) as connection:
        class_rows = connection.execute(
            "SELECT predicted_class, COUNT(*) AS total FROM prediction_results GROUP BY predicted_class"
        ).fetchall()
        type_rows = connection.execute(
            """SELECT t.transaction_type,
                      SUM(CASE WHEN p.predicted_class != 'Legitimate' THEN 1 ELSE 0 END) AS flagged,
                      COUNT(*) AS total
               FROM transactions t JOIN prediction_results p ON p.transaction_fk = t.id
               GROUP BY t.transaction_type ORDER BY t.transaction_type"""
        ).fetchall()
        trend_rows = connection.execute(
            """SELECT date(p.created_at) AS day,
                      SUM(CASE WHEN p.predicted_class != 'Legitimate' THEN 1 ELSE 0 END) AS flagged,
                      COUNT(*) AS total
               FROM prediction_results p GROUP BY date(p.created_at) ORDER BY day"""
        ).fetchall()
        amount_rows = connection.execute(
            """SELECT t.amount, p.predicted_class FROM transactions t
               JOIN prediction_results p ON p.transaction_fk = t.id ORDER BY t.amount"""
        ).fetchall()
        score_rows = connection.execute(
            "SELECT risk_score FROM prediction_results ORDER BY risk_score"
        ).fetchall()
    model_metrics = artifact["metrics"]
    return render_template(
        "analytics.html",
        class_data={row["predicted_class"]: row["total"] for row in class_rows},
        type_data=[dict(row) for row in type_rows],
        trend_data=[dict(row) for row in trend_rows],
        amount_data=[dict(row) for row in amount_rows],
        score_data=[row["risk_score"] for row in score_rows],
        metrics=model_metrics,
    )


@main.route("/model")
def model_info():
    artifact = current_app.extensions["fraud_model"]
    with open(current_app.config["DATA_DIR"] + "/eda_summary.json", encoding="utf-8") as summary_file:
        eda_summary = json.load(summary_file)
    classifier = artifact["pipeline"].named_steps["classifier"]
    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    encoded_names = preprocessor.get_feature_names_out()
    raw_importances = classifier.feature_importances_
    aggregated = {name: 0.0 for name in FEATURES}
    for encoded, importance in zip(encoded_names, raw_importances):
        raw_name = encoded.split("__", 1)[-1]
        feature = next((name for name in FEATURES if raw_name == name or raw_name.startswith(f"{name}_")), raw_name)
        aggregated[feature] = aggregated.get(feature, 0.0) + float(importance)
    importances = sorted(
        [{"name": name.replace("_", " ").title(), "value": round(value, 4)} for name, value in aggregated.items()],
        key=lambda item: item["value"], reverse=True,
    )
    return render_template(
        "model.html", metrics=artifact["metrics"], confusion=artifact["confusion_matrix"],
        importances=importances, version=artifact["version"], eda_summary=eda_summary,
    )


@main.route("/api/analytics")
def analytics_api():
    with closing(_database()) as connection:
        rows = connection.execute(
            "SELECT risk_level, COUNT(*) AS total FROM prediction_results GROUP BY risk_level"
        ).fetchall()
    return jsonify({row["risk_level"]: row["total"] for row in rows})


@main.route("/eda/<filename>")
def eda_asset(filename):
    if filename not in {"eda_class_and_amount.png", "eda_type_rates.png"}:
        abort(404)
    return send_from_directory(current_app.config["DATA_DIR"], filename)