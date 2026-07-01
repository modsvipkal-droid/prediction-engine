import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import settings


class PredictionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class SelfLearningEngine:
    def __init__(self):
        self.model: PredictionModel | None = None
        self.input_size: int = settings.LAG_SEQUENCE_SIZE
        self.hidden_size: int = settings.HIDDEN_SIZE
        self.accuracy: float = 0.0
        self.loss: float = 0.0
        self.samples_trained: int = 0
        self.latest_prediction: dict = {}
        self.pattern_memory: dict = {}
        self.strategy_notes: list = []
        self.total_red: int = 0
        self.total_green: int = 0
        self.total_data: int = 0
        self.hourly_patterns: dict = {}
        self.streak_data: dict = {}
        self.last_period: str = ""

    def _encode_result(self, r: str) -> float:
        return 1.0 if r == "Green" else 0.0

    def _decode_result(self, val: float) -> str:
        return "Green" if val >= 0.5 else "Red"

    def _build_features(self, documents: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        results = np.array([self._encode_result(d["result"]) for d in documents])
        periods = [d["period"] for d in documents]
        X, y = [], []
        for i in range(len(results) - self.input_size):
            seq = results[i : i + self.input_size]
            seq_features = self._add_time_features(seq, periods[i : i + self.input_size])
            X.append(seq_features)
            y.append(results[i + self.input_size])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _add_time_features(self, seq: np.ndarray, periods: list[str]) -> np.ndarray:
        hour = int(periods[-1][8:10]) / 23.0 if periods[-1] else 0.5
        streak = self._compute_current_streak(seq)
        recent_avg = float(np.mean(seq[-5:])) if len(seq) >= 5 else 0.5
        green_ratio = float(np.sum(seq) / len(seq)) if len(seq) > 0 else 0.5
        return np.append(seq, [hour, streak, recent_avg, green_ratio])

    def _compute_current_streak(self, seq: np.ndarray) -> float:
        if len(seq) == 0:
            return 0.0
        last = seq[-1]
        count = 0
        for v in reversed(seq):
            if v == last:
                count += 1
            else:
                break
        return count / 10.0

    def _extract_strategy(self) -> list[str]:
        strategies = []
        if self.hourly_patterns:
            for hr, info in sorted(self.hourly_patterns.items()):
                if info["dominant"] != "Equal" and info["total"] >= 10:
                    strategies.append(
                        f"Hour {hr}: {info['dominant']} dominates ({info.get(info['dominant'].lower() + '_pct', 0)}%)"
                    )
        if self.streak_data:
            if self.streak_data.get("avg_streak_red", 0) > 3:
                strategies.append(
                    f"Red has long streaks (avg {self.streak_data['avg_streak_red']})"
                )
            if self.streak_data.get("avg_streak_green", 0) > 3:
                strategies.append(
                    f"Green has long streaks (avg {self.streak_data['avg_streak_green']})"
                )
        if not strategies:
            strategies.append("No dominant hourly pattern detected")
        return strategies

    def train(self, documents: list[dict], hourly: dict, streaks: dict):
        self.total_data = len(documents)
        self.total_red = sum(1 for d in documents if d["result"] == "Red")
        self.total_green = sum(1 for d in documents if d["result"] == "Green")
        self.hourly_patterns = hourly
        self.streak_data = streaks
        if documents:
            self.last_period = documents[-1]["period"]

        if len(documents) < self.input_size + 2:
            self.latest_prediction = {
                "predicted_result": "Red",
                "confidence": 0.5,
                "probabilities": {"Red": 0.5, "Green": 0.5},
            }
            return

        X, y = self._build_features(documents)
        n_samples = len(X)
        split = int(n_samples * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        input_dim = X.shape[1]
        self.model = PredictionModel(input_dim, self.hidden_size)

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)

        self.model.train()
        for epoch in range(settings.EPOCHS):
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.loss = running_loss / len(train_loader)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test))
            pred_binary = (preds.numpy() >= 0.5).astype(int).flatten()
            actual = y_test.astype(int)
            self.accuracy = float(np.mean(pred_binary == actual))

        self.samples_trained = n_samples
        self.strategy_notes = self._extract_strategy()
        self._predict_next(X, documents)

    def _predict_next(self, X: np.ndarray, documents: list[dict]):
        if self.model is None:
            return
        last_seq = X[-1:].copy()
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(last_seq))
            green_prob = float(output.numpy().flatten()[0])
            red_prob = 1.0 - green_prob

        predicted = "Green" if green_prob >= 0.5 else "Red"
        confidence = max(green_prob, red_prob)

        current_hour = documents[-1]["period"][8:10] if documents else "??"
        hour_info = self.hourly_patterns.get(current_hour, {})
        hour_note = ""
        if hour_info:
            hour_note = f"Hour {current_hour}: {hour_info['dominant']} {hour_info.get(hour_info['dominant'].lower() + '_pct', 0)}%"

        seq_len = len(self.last_period)
        suffix_len = 5
        seq_part = self.last_period[:seq_len - suffix_len]
        seq_num = int(self.last_period[-suffix_len:]) if suffix_len <= seq_len else 1
        next_period = seq_part + str(seq_num + 1).zfill(suffix_len) if self.last_period else f"N1"

        self.latest_prediction = {
            "current_period": self.last_period,
            "next_period": next_period,
            "prediction": predicted,
            "confidence": round(confidence, 4),
            "probabilities": {"Red": round(red_prob, 4), "Green": round(green_prob, 4)},
            "hourly_pattern": hour_note,
            "strategy": self.strategy_notes,
            "total_data": self.total_data,
            "total_red": self.total_red,
            "total_green": self.total_green,
            "accuracy": round(self.accuracy, 4),
            "loss": round(self.loss, 4),
            "samples_trained": self.samples_trained,
        }


engine = SelfLearningEngine()
