import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import settings


class PredictionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        if hidden_size >= 2:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(hidden_size, max(hidden_size // 2, 2)))
        layers.append(nn.ReLU())
        if hidden_size // 2 >= 2:
            layers.append(nn.BatchNorm1d(max(hidden_size // 2, 2)))
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(max(hidden_size // 2, 2), max(hidden_size // 4, 2)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(max(hidden_size // 4, 2), 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

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
        self.latest_prediction: dict = self._default_prediction()
        self.strategy_notes: list = ["No data yet"]
        self.total_red: int = 0
        self.total_green: int = 0
        self.total_data: int = 0
        self.hourly_patterns: dict = {}
        self.streak_data: dict = {}
        self.last_period: str = ""

    def _default_prediction(self) -> dict:
        return {
            "current_period": "",
            "next_period": "",
            "prediction": "Red",
            "confidence": 0.0,
            "probabilities": {"Red": 0.5, "Green": 0.5},
            "hourly_pattern": "No hourly data yet",
            "strategy": ["Analyzing patterns..."],
            "total_data": 0,
            "total_red": 0,
            "total_green": 0,
            "accuracy": 0.0,
            "loss": 0.0,
            "samples_trained": 0,
        }

    def _encode_result(self, r: str) -> float:
        return 1.0 if r == "Green" else 0.0

    def _build_features(self, documents: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        results = np.array([self._encode_result(d["result"]) for d in documents], dtype=np.float32)
        periods = [d.get("period", "") for d in documents]
        X, y = [], []
        n = len(results)
        for i in range(n - self.input_size):
            seq = results[i : i + self.input_size]
            seq_features = self._add_time_features(seq, periods[i : i + self.input_size])
            X.append(seq_features)
            y.append(results[i + self.input_size])
        if not X:
            return np.empty((0, self.input_size + 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _add_time_features(self, seq: np.ndarray, periods: list[str]) -> np.ndarray:
        hour = 0.5
        if periods and periods[-1] and len(periods[-1]) >= 10:
            try:
                hour = int(periods[-1][8:10]) / 23.0
            except ValueError:
                hour = 0.5
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
        return count / float(self.input_size)

    def _extract_strategy(self) -> list[str]:
        strategies = []
        if self.hourly_patterns:
            for hr, info in sorted(self.hourly_patterns.items()):
                if info["dominant"] != "Equal" and info["total"] >= 5:
                    pct = info.get(f"{info['dominant'].lower()}_pct", 0)
                    strategies.append(f"Hour {hr}: {info['dominant']} {pct}%")
        if self.streak_data:
            ar = self.streak_data.get("avg_streak_red", 0)
            ag = self.streak_data.get("avg_streak_green", 0)
            if ar > 2:
                strategies.append(f"Red avg streak: {ar}")
            if ag > 2:
                strategies.append(f"Green avg streak: {ag}")
        return strategies if strategies else ["Balanced pattern, no strong hourly edge"]

    def train(self, documents: list[dict], hourly: dict, streaks: dict):
        self.total_data = len(documents)
        self.total_red = sum(1 for d in documents if d["result"] == "Red")
        self.total_green = sum(1 for d in documents if d["result"] == "Green")
        self.hourly_patterns = hourly
        self.streak_data = streaks
        if documents:
            self.last_period = documents[-1].get("period", "")

        if len(documents) < self.input_size + 2:
            self.latest_prediction = {**self._default_prediction(), "strategy": ["Not enough data"]}
            return

        X, y = self._build_features(documents)
        n_samples = len(X)
        if n_samples < 2:
            self.latest_prediction = {**self._default_prediction(), "strategy": ["Not enough sequences"]}
            return

        split = max(int(n_samples * 0.8), 1)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        input_dim = X.shape[1]
        self.model = PredictionModel(input_dim, self.hidden_size)
        self.model.train()

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=min(settings.BATCH_SIZE, len(X_train)), shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)

        for _ in range(settings.EPOCHS):
            running_loss = 0.0
            batches = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batches += 1
            self.loss = running_loss / max(batches, 1)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_test))
            pred_binary = (preds.numpy() >= 0.5).astype(int).flatten()
            actual = y_test.astype(int)
            self.accuracy = float(np.mean(pred_binary == actual)) if len(actual) > 0 else 0.0

        self.samples_trained = n_samples
        self.strategy_notes = self._extract_strategy()
        self._predict_next(X, documents)

    def _predict_next(self, X: np.ndarray, documents: list[dict]):
        if self.model is None or len(X) == 0:
            return
        last_seq = X[-1:].copy()
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(last_seq))
            green_prob = float(output.numpy().flatten()[0])
            red_prob = 1.0 - green_prob

        predicted = "Green" if green_prob >= 0.5 else "Red"
        confidence = max(green_prob, red_prob)

        current_hour = "??"
        try:
            current_hour = documents[-1]["period"][8:10] if documents and "period" in documents[-1] else "??"
        except Exception:
            pass
        hour_info = self.hourly_patterns.get(current_hour, {})
        hour_note = f"No data for hour {current_hour}"
        if hour_info:
            dom = hour_info.get("dominant", "Equal")
            pct = hour_info.get(f"{dom.lower()}_pct", 0)
            hour_note = f"Hour {current_hour}: {dom} {pct}%"

        seq_len = len(self.last_period)
        suffix_len = 5
        next_period = ""
        if seq_len > suffix_len:
            try:
                seq_part = self.last_period[:seq_len - suffix_len]
                seq_num = int(self.last_period[-suffix_len:])
                next_period = seq_part + str(seq_num + 1).zfill(suffix_len)
            except ValueError:
                next_period = f"{self.last_period}_next"

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
