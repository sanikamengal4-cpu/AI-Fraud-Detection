import json
import sqlite3
from contextlib import closing


def connect_database(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(path):
    with closing(connect_database(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL UNIQUE,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL,
                transaction_hour INTEGER NOT NULL,
                sender_previous_balance REAL NOT NULL,
                sender_new_balance REAL NOT NULL,
                receiver_previous_balance REAL NOT NULL,
                receiver_new_balance REAL NOT NULL,
                transactions_last_hour INTEGER NOT NULL,
                device_type TEXT NOT NULL,
                location TEXT NOT NULL,
                account_age_days INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_fk INTEGER NOT NULL UNIQUE REFERENCES transactions(id) ON DELETE CASCADE,
                predicted_class TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                factors_json TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS model_evaluation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                confusion_matrix_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_transactions_created ON transactions(created_at);
            CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);
            CREATE INDEX IF NOT EXISTS idx_predictions_risk ON prediction_results(risk_level);
            """
        )
        connection.commit()


def save_prediction(path, transaction, result):
    with closing(connect_database(path)) as connection:
        cursor = connection.execute(
            """INSERT INTO transactions (
                transaction_id, amount, transaction_type, transaction_hour,
                sender_previous_balance, sender_new_balance, receiver_previous_balance,
                receiver_new_balance, transactions_last_hour, device_type, location,
                account_age_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                transaction["transaction_id"], transaction["amount"], transaction["transaction_type"],
                transaction["transaction_hour"], transaction["sender_previous_balance"],
                transaction["sender_new_balance"], transaction["receiver_previous_balance"],
                transaction["receiver_new_balance"], transaction["transactions_last_hour"],
                transaction["device_type"], transaction["location"], transaction["account_age_days"],
            ),
        )
        connection.execute(
            """INSERT INTO prediction_results
               (transaction_fk, predicted_class, risk_score, risk_level, factors_json, model_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                cursor.lastrowid,
                result["predicted_class"],
                result["risk_score"],
                result["risk_level"],
                json.dumps(result["factors"]),
                result["model_version"],
            ),
        )
        connection.commit()
        return cursor.lastrowid


def save_evaluation(path, metrics, confusion_matrix):
    with closing(connect_database(path)) as connection:
        connection.execute(
            "INSERT INTO model_evaluation_history (model_name, metrics_json, confusion_matrix_json) VALUES (?, ?, ?)",
            ("Random Forest Classifier", json.dumps(metrics), json.dumps(confusion_matrix)),
        )
        connection.commit()


def transaction_detail(path, transaction_pk):
    with closing(connect_database(path)) as connection:
        return connection.execute(
            """SELECT t.*, p.predicted_class, p.risk_score, p.risk_level,
                      p.factors_json, p.model_version, p.created_at AS predicted_at
               FROM transactions t JOIN prediction_results p ON p.transaction_fk = t.id
               WHERE t.id = ?""",
            (transaction_pk,),
        ).fetchone()