import os
from dotenv import load_dotenv

load_dotenv(override=False)


class Settings:
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "wingo_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "results_30s")
    LAG_SEQUENCE_SIZE: int = int(os.getenv("LAG_SEQUENCE_SIZE", "10"))
    TRAIN_INTERVAL_SECONDS: int = int(os.getenv("TRAIN_INTERVAL_SECONDS", "30"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.01"))
    EPOCHS: int = int(os.getenv("EPOCHS", "50"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    HIDDEN_SIZE: int = int(os.getenv("HIDDEN_SIZE", "64"))
    MAX_RECORDS: int = int(os.getenv("MAX_RECORDS", "5000"))


settings = Settings()
