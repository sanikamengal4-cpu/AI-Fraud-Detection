from pathlib import Path

from flask import Flask

from .database import initialize_database
from .model import load_or_train_model
from .routes import main


def create_app(test_config=None):
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.from_mapping(
        SECRET_KEY="educational-fraud-detector-change-me",
        DATABASE=str(data_dir / "fraud_detector.db"),
        DATA_DIR=str(data_dir),
        MODEL_PATH=str(data_dir / "fraud_pipeline.joblib"),
        DATASET_PATH=str(data_dir / "sample_transactions.csv"),
        PER_PAGE=10,
    )
    if test_config:
        app.config.update(test_config)

    initialize_database(app.config["DATABASE"])
    app.extensions["fraud_model"] = load_or_train_model(
        app.config["MODEL_PATH"],
        app.config["DATASET_PATH"],
        app.config["DATABASE"],
    )
    app.register_blueprint(main)

    return app