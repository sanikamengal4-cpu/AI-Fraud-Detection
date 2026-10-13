import tempfile
import unittest
from pathlib import Path

from fraud_detector import create_app
from fraud_detector.model import predict_transaction


class FraudDetectorAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temporary_directory.name)
        self.config = {
            "TESTING": True,
            "DATABASE": str(self.data_path / "test.db"),
            "DATA_DIR": str(self.data_path),
            "MODEL_PATH": str(self.data_path / "model.joblib"),
            "DATASET_PATH": str(self.data_path / "sample.csv"),
            "PER_PAGE": 2,
        }
        self.app = create_app(self.config)
        self.client = self.app.test_client()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_all_primary_pages_render(self):
        for path in ["/", "/analyze", "/history", "/analytics", "/model", "/api/analytics"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)
        for path in ["/eda/eda_class_and_amount.png", "/eda/eda_type_rates.png"]:
            image_response = self.client.get(path)
            self.assertEqual(image_response.status_code, 200)
            image_response.close()
        self.assertEqual(self.client.get("/eda/../sample.csv").status_code, 404)

    def test_submission_is_saved_and_survives_app_recreation(self):
        response = self.client.post("/analyze", data=self._transaction(), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"TX-TEST-001", response.data)
        self.assertIn(b"Signals that shifted this score", response.data)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/analytics").status_code, 200)

        restarted_app = create_app(self.config)
        history_response = restarted_app.test_client().get("/history?q=TX-TEST-001")
        self.assertEqual(history_response.status_code, 200)
        self.assertIn(b"TX-TEST-001", history_response.data)
        self.assertEqual(restarted_app.test_client().get("/analytics").status_code, 200)

    def test_filters_and_duplicate_ids_are_handled(self):
        self.client.post("/analyze", data=self._transaction())
        response = self.client.get("/history?transaction_type=CASH_OUT&risk_level=High")
        self.assertEqual(response.status_code, 200)
        duplicate = self.client.post("/analyze", data=self._transaction(), follow_redirects=True)
        self.assertIn(b"already exists", duplicate.data)

    def test_invalid_hour_is_rejected_without_saving(self):
        transaction = self._transaction()
        transaction["transaction_hour"] = "27"
        response = self.client.post("/analyze", data=transaction, follow_redirects=True)
        self.assertIn(b"between 0 and 23", response.data)
        self.assertIn(b"No transactions analyzed yet", self.client.get("/").data)

    def test_model_metrics_and_examples_have_meaningful_signal(self):
        artifact = self.app.extensions["fraud_model"]
        self.assertGreater(artifact["metrics"]["recall"], 0)
        routine = {
            "amount": 84.5,
            "transaction_type": "PAYMENT",
            "transaction_hour": 13,
            "sender_previous_balance": 4200.0,
            "sender_new_balance": 4115.5,
            "receiver_previous_balance": 1800.0,
            "receiver_new_balance": 1884.5,
            "transactions_last_hour": 1,
            "device_type": "Mobile",
            "location": "Local",
            "account_age_days": 740,
        }
        elevated = {
            **routine,
            "amount": 78400.0,
            "transaction_type": "CASH_OUT",
            "transaction_hour": 2,
            "sender_previous_balance": 79100.0,
            "sender_new_balance": 700.0,
            "receiver_previous_balance": 120.0,
            "receiver_new_balance": 78520.0,
            "transactions_last_hour": 9,
            "device_type": "Unknown",
            "location": "International",
            "account_age_days": 12,
        }
        routine_score = predict_transaction(artifact, routine)["risk_score"]
        elevated_score = predict_transaction(artifact, elevated)["risk_score"]
        self.assertGreater(elevated_score, routine_score)

    @staticmethod
    def _transaction():
        return {
            "transaction_id": "TX-TEST-001",
            "amount": "78400",
            "transaction_type": "CASH_OUT",
            "transaction_hour": "2",
            "sender_previous_balance": "79100",
            "sender_new_balance": "700",
            "receiver_previous_balance": "120",
            "receiver_new_balance": "78520",
            "transactions_last_hour": "9",
            "device_type": "Unknown",
            "location": "International",
            "account_age_days": "12",
        }


if __name__ == "__main__":
    unittest.main()