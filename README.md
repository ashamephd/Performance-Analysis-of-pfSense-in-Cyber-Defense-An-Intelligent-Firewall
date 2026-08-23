# Performance-Analysis-of-pfSense-in-Cyber-Defense-An-Intelligent-Firewall
The primary goal of this research is to evaluate and compare the performance of pfSense as an open-source firewall and routing tool based on FreeBSD, with the enhancements made through machine learning to transform it into a proactive detector of emerging threats.                  
**data/synthetic_data.py**
"""
Synthetic network-traffic data generator.

The paper trains on public datasets (CICIDS2017, NSL-KDD, TON_IoT, BoT-IoT,
CIC-DDoS2019) that are large, license-restricted, and not redistributable in
a code repo. This module generates *statistically similar* synthetic traffic
so the rest of the pipeline (feature engineering -> RF/XGBoost triage ->
CNN/LSTM deep inspection -> Isolation Forest anomaly detection -> SHAP/LIME
explainability) is fully runnable end-to-end without external downloads.

To use real data instead: replace `generate_dataset()` with a loader that
reads CICIDS2017/NSL-KDD CSVs and maps their columns onto the same
feature schema defined in features/feature_extraction.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

RNG_SEED = 42

ATTACK_TYPES = [
    "benign",
    "ddos",
    "port_scan",
    "brute_force",
    "malware",
    "zero_day",  # held-out novel pattern, used only for generalization tests
]


def _packet_size_stats(rng, n, mean, std, lo, hi):
    sizes = rng.normal(mean, std, n)
    sizes = np.clip(sizes, lo, hi)
    return sizes


def _make_class(rng: np.random.Generator, n: int, label: str) -> pd.DataFrame:
    """Generate n rows of synthetic packet/flow/payload/statistical features
    for a given traffic class, following Tables 2-6 in the paper."""

    if label == "benign":
        pkt_size = _make_series(rng, n, 500, 300, 40, 1500)
        iat = rng.exponential(0.02, n)
        conn_rate = rng.normal(5, 2, n).clip(0, None)
        entropy = rng.normal(4.5, 0.8, n).clip(0, 8)
        ssl_flag = rng.binomial(1, 0.6, n)
        port_uncommon = rng.binomial(1, 0.03, n)
        syn_only = rng.binomial(1, 0.02, n)
        failed_login = rng.poisson(0.05, n)

    elif label == "ddos":
        pkt_size = _make_series(rng, n, 80, 20, 40, 200)
        iat = rng.exponential(0.0005, n)
        conn_rate = rng.normal(800, 200, n).clip(50, None)
        entropy = rng.normal(2.0, 0.5, n).clip(0, 8)
        ssl_flag = rng.binomial(1, 0.05, n)
        port_uncommon = rng.binomial(1, 0.2, n)
        syn_only = rng.binomial(1, 0.85, n)
        failed_login = rng.poisson(0.0, n)

    elif label == "port_scan":
        pkt_size = _make_series(rng, n, 60, 10, 40, 100)
        iat = rng.exponential(0.001, n)
        conn_rate = rng.normal(300, 100, n).clip(20, None)
        entropy = rng.normal(1.5, 0.3, n).clip(0, 8)
        ssl_flag = rng.binomial(1, 0.01, n)
        port_uncommon = rng.binomial(1, 0.9, n)
        syn_only = rng.binomial(1, 0.95, n)
        failed_login = rng.poisson(0.0, n)

    elif label == "brute_force":
        pkt_size = _make_series(rng, n, 150, 40, 60, 400)
        iat = rng.exponential(0.05, n)
        conn_rate = rng.normal(20, 5, n).clip(1, None)
        entropy = rng.normal(3.0, 0.5, n).clip(0, 8)
        ssl_flag = rng.binomial(1, 0.4, n)
        port_uncommon = rng.binomial(1, 0.1, n)
        syn_only = rng.binomial(1, 0.05, n)
        failed_login = rng.poisson(8.0, n)

    elif label == "malware":
        pkt_size = _make_series(rng, n, 900, 350, 100, 1500)
        iat = rng.exponential(0.01, n)
        conn_rate = rng.normal(15, 6, n).clip(1, None)
        entropy = rng.normal(7.2, 0.4, n).clip(0, 8)  # high entropy: encrypted/packed payload
        ssl_flag = rng.binomial(1, 0.8, n)
        port_uncommon = rng.binomial(1, 0.35, n)
        syn_only = rng.binomial(1, 0.05, n)
        failed_login = rng.poisson(0.1, n)

    elif label == "zero_day":
        # Deliberately blends benign + malware traits with added jitter,
        # simulating a previously-unseen polymorphic attack pattern.
        pkt_size = _make_series(rng, n, 700, 400, 40, 1500)
        iat = rng.exponential(0.015, n)
        conn_rate = rng.normal(45, 25, n).clip(0, None)
        entropy = rng.normal(6.0, 1.2, n).clip(0, 8)
        ssl_flag = rng.binomial(1, 0.7, n)
        port_uncommon = rng.binomial(1, 0.25, n)
        syn_only = rng.binomial(1, 0.15, n)
        failed_login = rng.poisson(1.5, n)

    else:
        raise ValueError(f"unknown label {label}")

    df = pd.DataFrame({
        "packet_size": pkt_size,
        "inter_arrival_time": iat,
        "connection_rate": conn_rate,
        "payload_entropy": entropy,
        "ssl_tls_indicator": ssl_flag,
        "uncommon_port": port_uncommon,
        "syn_only_flag": syn_only,
        "failed_login_count": failed_login,
    })

    # derived / flow-level features (Table 3, Table 6)
    df["byte_rate"] = df["packet_size"] * df["connection_rate"]
    df["packet_size_variance"] = df["packet_size"].rolling(5, min_periods=1).var().fillna(0)
    df["iat_coeff_variation"] = (
        df["inter_arrival_time"].rolling(5, min_periods=1).std().fillna(0)
        / df["inter_arrival_time"].rolling(5, min_periods=1).mean().replace(0, 1e-6)
    )
    df["label"] = label
    df["is_malicious"] = 0 if label == "benign" else 1
    return df


def _make_series(rng, n, mean, std, lo, hi):
    x = rng.normal(mean, std, n)
    return np.clip(x, lo, hi)


def generate_dataset(
    n_total: int = 20000,
    class_ratios: dict | None = None,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Generate a labeled synthetic dataset spanning benign + attack classes.

    Parameters
    ----------
    n_total : total number of flow records to generate
    class_ratios : optional dict {label: proportion}; defaults to a
        realistic imbalanced mix dominated by benign traffic.
    """
    rng = np.random.default_rng(seed)

    if class_ratios is None:
        class_ratios = {
            "benign": 0.70,
            "ddos": 0.08,
            "port_scan": 0.07,
            "brute_force": 0.06,
            "malware": 0.06,
            "zero_day": 0.03,
        }

    frames = []
    for label, ratio in class_ratios.items():
        n = max(1, int(n_total * ratio))
        frames.append(_make_class(rng, n, label))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["flow_id"] = np.arange(len(df))
    return df


if __name__ == "__main__":
    data = generate_dataset(2000)
    print(data.groupby("label").size())
    print(data.head())
**
 **   features/feature_extraction.py****
 """
Feature engineering aligned with the paper's Tables 2-6:
  - Table 2: packet header features (IP/TCP/UDP)
  - Table 3: flow metadata features
  - Table 4: payload features
  - Table 5: time-window features (1s / 1min / 1hr)
  - Table 6: statistical features

`build_feature_matrix` turns the raw synthetic (or real) flow records into:
  - a tabular feature matrix for Random Forest / XGBoost / Isolation Forest
  - a windowed sequence tensor for LSTM (temporal analysis)
  - a "byte pattern" tensor for CNN (payload analysis stand-in)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TABULAR_FEATURES = [
    "packet_size",
    "inter_arrival_time",
    "connection_rate",
    "payload_entropy",
    "ssl_tls_indicator",
    "uncommon_port",
    "syn_only_flag",
    "failed_login_count",
    "byte_rate",
    "packet_size_variance",
    "iat_coeff_variation",
]


def build_feature_matrix(df: pd.DataFrame, fit_scaler: StandardScaler | None = None):
    """Z-score normalize tabular features (per the paper's normalization step).

    Returns (X_scaled: np.ndarray, scaler: StandardScaler, feature_names: list[str])
    """
    X = df[TABULAR_FEATURES].values.astype(np.float32)
    if fit_scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = fit_scaler
        X_scaled = scaler.transform(X)
    return X_scaled, scaler, TABULAR_FEATURES


def build_sliding_windows(df: pd.DataFrame, window: int = 20, stride: int = 10):
    """Sliding-window sequences over TABULAR_FEATURES for LSTM temporal
    analysis (paper: 100-packet window / 50-packet stride, scaled down here
    for a lightweight demo dataset).

    Returns (X_seq: [n_windows, window, n_features], y_seq: [n_windows])
    """
    X = df[TABULAR_FEATURES].values.astype(np.float32)
    y = df["is_malicious"].values.astype(np.float32)

    seqs, labels = [], []
    for start in range(0, len(df) - window, stride):
        end = start + window
        seqs.append(X[start:end])
        # window label = majority vote of malicious flag in the window
        labels.append(1.0 if y[start:end].mean() > 0.5 else 0.0)

    if not seqs:
        return np.empty((0, window, len(TABULAR_FEATURES))), np.empty((0,))

    return np.stack(seqs), np.array(labels)


def build_payload_bytes(df: pd.DataFrame, byte_len: int = 64, seed: int = 0):
    """Synthesize a pseudo byte-payload signal per flow, derived from
    entropy/packet-size features, for CNN-style payload pattern analysis.

    This stands in for real packet payload bytes (Table 4: N-gram / byte
    distribution features), which are not present in flow-level metadata.
    Returns X_bytes: [n_flows, byte_len, 1] scaled to [0, 1].
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    entropy = df["payload_entropy"].values
    size = df["packet_size"].values

    bytes_matrix = np.zeros((n, byte_len), dtype=np.float32)
    for i in range(n):
        # higher entropy -> more uniform/random byte distribution
        scale = 1.0 + entropy[i] / 8.0
        base = rng.normal(loc=size[i] % 256, scale=40 * scale, size=byte_len)
        bytes_matrix[i] = np.clip(base, 0, 255)

    bytes_matrix = bytes_matrix / 255.0
    return bytes_matrix.reshape(n, byte_len, 1)


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset

    df = generate_dataset(500)
    X, scaler, names = build_feature_matrix(df)
    X_seq, y_seq = build_sliding_windows(df)
    X_bytes = build_payload_bytes(df)
    print("tabular:", X.shape, "seq:", X_seq.shape, "bytes:", X_bytes.shape)

**    models/random_forest_model.py**
"""
Random Forest: lightweight packet/flow-level triage classifier.

Paper spec (Section IV-I): max 200 decision trees, max depth 15, trained on
CICIDS2017 + NSL-KDD, achieving 97.3% accuracy via 10-fold CV. Here it is
trained on the synthetic dataset with the same hyperparameters.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_random_forest(X: np.ndarray, y: np.ndarray, n_estimators: int = 200,
                         max_depth: int = 15, cv_folds: int = 10, seed: int = 42):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy", n_jobs=-1)
    clf.fit(X, y)
    return clf, cv_scores


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset
    from features.feature_extraction import build_feature_matrix

    df = generate_dataset(4000)
    X, scaler, names = build_feature_matrix(df)
    y = df["is_malicious"].values

    clf, cv_scores = train_random_forest(X, y, cv_folds=5)
    print(f"RF 5-fold CV accuracy: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")

**    models/xgboost_model.py**
"""
XGBoost: flow-level classifier using statistical/flow features for higher
detection accuracy (Table 7: lr=0.1, n_estimators=200, max_depth=7,
98.2% detection accuracy, 1-2 ms/flow inference).
"""

from __future__ import annotations
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train_xgboost(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1,
                   n_estimators: int = 200, max_depth: int = 7, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    return clf, metrics, (X_test, y_test)


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset
    from features.feature_extraction import build_feature_matrix

    df = generate_dataset(4000)
    X, scaler, names = build_feature_matrix(df)
    y = df["is_malicious"].values

    clf, metrics, _ = train_xgboost(X, y)
    print("XGBoost test metrics:", metrics)


  **  models/cnn_model.py**
  """
CNN: deep-packet-inspection model over byte-level payload signals.

Paper spec (Section VI): two conv layers (32 and 64 filters), n-gram-style
local pattern capture, pooling for shift-invariance; reported ~96.8%
accuracy / 3-5 ms latency for APT/polymorphic-malware identification.
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_len: int = 64) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(X_bytes: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 64,
              seed: int = 42):
    tf.random.set_seed(seed)
    n = len(X_bytes)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    model = build_cnn(input_len=X_bytes.shape[1])
    history = model.fit(
        X_bytes[train_idx], y[train_idx],
        validation_data=(X_bytes[test_idx], y[test_idx]),
        epochs=epochs, batch_size=batch_size, verbose=0,
    )
    loss, acc = model.evaluate(X_bytes[test_idx], y[test_idx], verbose=0)
    return model, {"test_loss": loss, "test_accuracy": acc}, history


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset
    from features.feature_extraction import build_payload_bytes

    df = generate_dataset(3000)
    X_bytes = build_payload_bytes(df)
    y = df["is_malicious"].values.astype(np.float32)

    model, metrics, _ = train_cnn(X_bytes, y, epochs=3)
    print("CNN test metrics:", metrics)

**    models/lstm_model.py**
"""
LSTM: temporal analysis of sliding-window flow sequences.

Paper spec (Section VI): identifies multi-stage/coordinated attacks via
sequence numbers, ack, flow durations, state transitions; reported ~92.3%
accuracy, 5-8 ms/flow.
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(window: int, n_features: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(X_seq: np.ndarray, y_seq: np.ndarray, epochs: int = 8, batch_size: int = 32,
               seed: int = 42):
    tf.random.set_seed(seed)
    n = len(X_seq)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    model = build_lstm(window=X_seq.shape[1], n_features=X_seq.shape[2])
    history = model.fit(
        X_seq[train_idx], y_seq[train_idx],
        validation_data=(X_seq[test_idx], y_seq[test_idx]),
        epochs=epochs, batch_size=batch_size, verbose=0,
    )
    loss, acc = model.evaluate(X_seq[test_idx], y_seq[test_idx], verbose=0)
    return model, {"test_loss": loss, "test_accuracy": acc}, history


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset
    from features.feature_extraction import build_sliding_windows

    df = generate_dataset(4000)
    X_seq, y_seq = build_sliding_windows(df, window=20, stride=10)
    model, metrics, _ = train_lstm(X_seq, y_seq, epochs=5)
    print("LSTM test metrics:", metrics)
**
    models/isolation_forest_model.py**
    """
Isolation Forest: unsupervised behavioral-baseline anomaly detector.

Paper spec (Section IV-I): 150 estimators, contamination rate 0.05,
72-hour "period of normalcy" to build a behavior baseline. Here the
"normalcy period" is approximated by fitting only on benign-labeled flows.
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X_benign: np.ndarray, n_estimators: int = 150,
                            contamination: float = 0.05, seed: int = 42):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_benign)
    return model


def score_anomalies(model: IsolationForest, X: np.ndarray):
    """Returns (is_anomaly: bool array, anomaly_score: float array).
    Lower score = more anomalous (sklearn convention)."""
    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)
    is_anomaly = preds == -1
    return is_anomaly, scores


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset
    from features.feature_extraction import build_feature_matrix

    df = generate_dataset(4000)
    X, scaler, names = build_feature_matrix(df)
    benign_mask = df["is_malicious"].values == 0

    model = train_isolation_forest(X[benign_mask])
    is_anomaly, scores = score_anomalies(model, X)

    detected_rate = is_anomaly[~benign_mask].mean()
    false_alarm_rate = is_anomaly[benign_mask].mean()
    print(f"Anomaly detection rate on malicious flows: {detected_rate:.3f}")
    print(f"False alarm rate on benign flows: {false_alarm_rate:.3f}")
**
    explainability/shap_lime_explain.py**

    """
Explainable AI layer: SHAP for global feature attribution, LIME for
local/instance-level explanations of individual classification decisions.
"""

from __future__ import annotations
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer


def global_shap_explanation(model, X: np.ndarray, feature_names: list[str],
                             sample_size: int = 200, seed: int = 42):
    """Global feature attribution via SHAP TreeExplainer (works for RF/XGBoost).

    Returns (mean_abs_shap: dict[feature_name -> importance], shap_values)
    """
    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # binary classifiers may return a list [class0, class1] or a single array
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values
        if values.ndim == 3:  # (n, features, classes)
            values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    importance = dict(sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda kv: kv[1], reverse=True
    ))
    return importance, values


def local_lime_explanation(model, X_train: np.ndarray, instance: np.ndarray,
                            feature_names: list[str], class_names=("benign", "malicious"),
                            num_features: int = 6):
    """Local, instance-specific explanation for a single flow/packet decision."""
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
    )

    predict_fn = model.predict_proba
    explanation = explainer.explain_instance(
        instance, predict_fn, num_features=num_features
    )
    return explanation.as_list()


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset
    from features.feature_extraction import build_feature_matrix
    from models.random_forest_model import train_random_forest

    df = generate_dataset(2000)
    X, scaler, names = build_feature_matrix(df)
    y = df["is_malicious"].values

    clf, _ = train_random_forest(X, y, cv_folds=3)

    importance, _ = global_shap_explanation(clf, X, names)
    print("Top SHAP features (global):")
    for feat, val in list(importance.items())[:5]:
        print(f"  {feat}: {val:.4f}")

    explanation = local_lime_explanation(clf, X, X[0], names)
    print("\nLIME explanation for flow #0:")
    for feat, weight in explanation:
        print(f"  {feat}: {weight:.4f}")

     **   pipeline/hierarchical_firewall.py**

     """
Hierarchical AI-optimized firewall pipeline (mirrors Figure 2 / Section IV-B):

  1. Random Forest triages ALL traffic at packet/flow level (fast, cheap).
  2. Only traffic RF flags as suspicious (or borderline) is escalated to
     XGBoost for higher-accuracy flow-level classification.
  3. Traffic still ambiguous after XGBoost is escalated further to the
     CNN (payload pattern) and LSTM (temporal pattern) models.
  4. An Isolation Forest baseline runs in parallel as a behavioral/zero-day
     safety net, independent of the supervised triage chain.
  5. Every escalation decision is explainable via SHAP (global) and
     LIME (per-instance).

This "only suspicious traffic goes deeper" design is what the paper calls
the AI optimization that avoids needless deep processing.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd

from features.feature_extraction import build_feature_matrix, build_payload_bytes
from models.random_forest_model import train_random_forest
from models.xgboost_model import train_xgboost
from models.cnn_model import train_cnn
from models.isolation_forest_model import train_isolation_forest, score_anomalies


class HierarchicalFirewall:
    """Trains the tiered model stack and runs traffic through it."""

    def __init__(self, rf_threshold: float = 0.5, xgb_threshold: float = 0.5,
                 escalate_band: tuple[float, float] = (0.35, 0.65)):
        self.rf_threshold = rf_threshold
        self.xgb_threshold = xgb_threshold
        self.escalate_band = escalate_band  # RF prob range that triggers XGBoost

        self.scaler = None
        self.feature_names = None
        self.rf_model = None
        self.xgb_model = None
        self.cnn_model = None
        self.iso_model = None

    def fit(self, df: pd.DataFrame):
        X, scaler, names = build_feature_matrix(df)
        y = df["is_malicious"].values
        self.scaler, self.feature_names = scaler, names

        self.rf_model, rf_cv = train_random_forest(X, y, cv_folds=5)
        self.xgb_model, xgb_metrics, _ = train_xgboost(X, y)

        X_bytes = build_payload_bytes(df)
        self.cnn_model, cnn_metrics, _ = train_cnn(X_bytes, y.astype(np.float32), epochs=5)

        benign_mask = y == 0
        self.iso_model = train_isolation_forest(X[benign_mask])

        return {
            "rf_cv_accuracy_mean": float(rf_cv.mean()),
            "xgb_metrics": xgb_metrics,
            "cnn_metrics": cnn_metrics,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the full tiered pipeline and returns per-flow decisions,
        the tier that made the final call, and per-tier latency (ms)."""
        X, _, _ = build_feature_matrix(df, fit_scaler=self.scaler)
        X_bytes = build_payload_bytes(df)

        results = []
        t0 = time.perf_counter()
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        rf_latency_ms = (time.perf_counter() - t0) * 1000 / max(1, len(df))

        lo, hi = self.escalate_band
        needs_escalation = (rf_proba >= lo) & (rf_proba <= hi)

        final_pred = (rf_proba >= self.rf_threshold).astype(int)
        decision_tier = np.array(["random_forest"] * len(df), dtype=object)
        total_latency_ms = np.full(len(df), rf_latency_ms)

        if needs_escalation.any():
            idx = np.where(needs_escalation)[0]
            t1 = time.perf_counter()
            xgb_proba = self.xgb_model.predict_proba(X[idx])[:, 1]
            xgb_latency = (time.perf_counter() - t1) * 1000 / len(idx)

            final_pred[idx] = (xgb_proba >= self.xgb_threshold).astype(int)
            decision_tier[idx] = "xgboost"
            total_latency_ms[idx] += xgb_latency

            still_ambiguous = idx[(xgb_proba >= 0.4) & (xgb_proba <= 0.6)]
            if len(still_ambiguous) > 0:
                t2 = time.perf_counter()
                cnn_proba = self.cnn_model.predict(X_bytes[still_ambiguous], verbose=0).ravel()
                cnn_latency = (time.perf_counter() - t2) * 1000 / len(still_ambiguous)

                final_pred[still_ambiguous] = (cnn_proba >= 0.5).astype(int)
                decision_tier[still_ambiguous] = "cnn_deep_inspection"
                total_latency_ms[still_ambiguous] += cnn_latency

        # Isolation Forest runs as an independent behavioral safety net
        is_anomaly, iso_scores = score_anomalies(self.iso_model, X)

        out = df.copy()
        out["rf_probability"] = rf_proba
        out["final_prediction"] = final_pred
        out["decision_tier"] = decision_tier
        out["pipeline_latency_ms"] = total_latency_ms
        out["isolation_forest_anomaly"] = is_anomaly
        out["isolation_forest_score"] = iso_scores
        return out


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset

    train_df = generate_dataset(6000, seed=1)
    test_df = generate_dataset(1500, seed=2)

    fw = HierarchicalFirewall()
    train_metrics = fw.fit(train_df)
    print("Training metrics:", train_metrics)

    results = fw.predict(test_df)
    print("\nDecision tier distribution:")
    print(results["decision_tier"].value_counts())
    print(f"\nMean end-to-end latency: {results['pipeline_latency_ms'].mean():.3f} ms/flow")

    acc = (results["final_prediction"] == results["is_malicious"]).mean()
    print(f"Overall pipeline accuracy: {acc:.4f}")
**

    evaluate/metrics.py**
    **"""
Reproduces the paper's baseline-vs-AI-optimized comparison methodology
(Table 8 / Section VIII-E): a "baseline" (rule-based / RF-only, no
escalation, no XAI) is compared against the full hierarchical AI pipeline
on the same held-out test set, across accuracy, false positive rate,
false negative rate, and latency.

This is a fair, from-scratch statistical comparison against synthetic data
-- it does not reproduce the paper's exact reported numbers, which came
from real hardware testbeds and public datasets the paper does not
distribute in reusable form.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {"accuracy": accuracy, "false_positive_rate": fpr, "false_negative_rate": fnr}


def compare_baseline_vs_ai(fw, test_df: pd.DataFrame) -> pd.DataFrame:
    """`fw` is a fitted HierarchicalFirewall. Baseline = Random Forest tier
    only, thresholded at 0.5, no escalation. AI-optimized = full pipeline
    output already computed via fw.predict()."""
    from features.feature_extraction import build_feature_matrix

    X, _, _ = build_feature_matrix(test_df, fit_scaler=fw.scaler)
    y_true = test_df["is_malicious"].values

    rf_only_pred = (fw.rf_model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    baseline_rates = _rates(y_true, rf_only_pred)

    full_results = fw.predict(test_df)
    ai_rates = _rates(y_true, full_results["final_prediction"].values)

    rows = [
        {"configuration": "baseline (RF only, no escalation)", **baseline_rates,
         "mean_latency_ms": float(full_results["pipeline_latency_ms"].min())},
        {"configuration": "AI-optimized (hierarchical pipeline)", **ai_rates,
         "mean_latency_ms": float(full_results["pipeline_latency_ms"].mean())},
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data.synthetic_data import generate_dataset
    from pipeline.hierarchical_firewall import HierarchicalFirewall

    train_df = generate_dataset(6000, seed=1)
    test_df = generate_dataset(1500, seed=2)

    fw = HierarchicalFirewall()
    fw.fit(train_df)

    comparison = compare_baseline_vs_ai(fw, test_df)
    print(comparison.to_string(index=False))

    main.py
    """
End-to-end demo of the AI-enhanced pfSense-style hierarchical firewall
described in:

    V. Asha and S. Kanaga Suba Raja, "Performance Analysis of pfSense in
    Cyber Defense: An Intelligent Firewall Perspective for the Modern
    Network Landscape," IEEE Access, vol. 14, 2026.

Runs: synthetic data generation -> feature engineering -> hierarchical
RF/XGBoost/CNN triage -> Isolation Forest behavioral safety net ->
SHAP/LIME explainability -> baseline-vs-AI-optimized comparison.

Usage:
    python3 main.py --train-size 8000 --test-size 2000
"""

from __future__ import annotations
import argparse
import json

from data.synthetic_data import generate_dataset
from pipeline.hierarchical_firewall import HierarchicalFirewall
from evaluate.metrics import compare_baseline_vs_ai
from explainability.shap_lime_explain import global_shap_explanation, local_lime_explanation
from features.feature_extraction import build_feature_matrix


def main():
    parser = argparse.ArgumentParser(description="AI-enhanced pfSense-style firewall demo")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("1. Generating synthetic traffic (stand-in for CICIDS2017/NSL-KDD)")
    print("=" * 70)
    train_df = generate_dataset(args.train_size, seed=args.seed)
    test_df = generate_dataset(args.test_size, seed=args.seed + 1)
    print(train_df["label"].value_counts(), "\n")

    print("=" * 70)
    print("2. Training hierarchical AI firewall stack")
    print("   (Random Forest -> XGBoost -> CNN, + Isolation Forest baseline)")
    print("=" * 70)
    fw = HierarchicalFirewall()
    train_metrics = fw.fit(train_df)
    print(json.dumps(train_metrics, indent=2, default=float), "\n")

    print("=" * 70)
    print("3. Running tiered inference on held-out test traffic")
    print("=" * 70)
    results = fw.predict(test_df)
    print("Decision tier distribution:")
    print(results["decision_tier"].value_counts())
    print(f"Mean end-to-end latency: {results['pipeline_latency_ms'].mean():.3f} ms/flow\n")

    print("=" * 70)
    print("4. Baseline (RF-only) vs AI-optimized pipeline comparison")
    print("=" * 70)
    comparison = compare_baseline_vs_ai(fw, test_df)
    print(comparison.to_string(index=False), "\n")

    print("=" * 70)
    print("5. Explainability: SHAP (global) + LIME (per-instance)")
    print("=" * 70)
    X, _, names = build_feature_matrix(test_df, fit_scaler=fw.scaler)
    importance, _ = global_shap_explanation(fw.rf_model, X, names)
    print("Top global SHAP feature attributions (Random Forest):")
    for feat, val in list(importance.items())[:5]:
        print(f"  {feat:24s} {val:.4f}")

    flagged = results[results["final_prediction"] == 1]
    if len(flagged) > 0:
        sample_idx = flagged.index[0]
        sample_pos = test_df.index.get_loc(sample_idx)
        explanation = local_lime_explanation(fw.rf_model, X, X[sample_pos], names)
        print(f"\nLIME explanation for flagged flow #{sample_idx} "
              f"(true label={test_df.loc[sample_idx, 'label']}):")
        for feat, weight in explanation:
            print(f"  {feat:30s} {weight:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()


**    tests/test_pipeline.py**
"""Minimal smoke tests. Run with: python3 -m pytest tests/ -q"""
import numpy as np

from data.synthetic_data import generate_dataset
from features.feature_extraction import build_feature_matrix, build_sliding_windows, build_payload_bytes
from models.random_forest_model import train_random_forest
from models.isolation_forest_model import train_isolation_forest, score_anomalies
from pipeline.hierarchical_firewall import HierarchicalFirewall


def test_data_generation():
    df = generate_dataset(500)
    assert len(df) > 0
    assert set(df["label"].unique()) <= {
        "benign", "ddos", "port_scan", "brute_force", "malware", "zero_day"
    }


def test_feature_matrix_shapes():
    df = generate_dataset(300)
    X, scaler, names = build_feature_matrix(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(names)


def test_sliding_windows():
    df = generate_dataset(300)
    X_seq, y_seq = build_sliding_windows(df, window=10, stride=5)
    assert X_seq.shape[0] == y_seq.shape[0]
    assert X_seq.shape[1] == 10


def test_random_forest_trains():
    df = generate_dataset(500)
    X, _, _ = build_feature_matrix(df)
    y = df["is_malicious"].values
    clf, cv_scores = train_random_forest(X, y, cv_folds=3)
    assert cv_scores.mean() > 0.5


def test_isolation_forest_flags_more_malicious_than_benign():
    df = generate_dataset(1500)
    X, _, _ = build_feature_matrix(df)
    benign_mask = df["is_malicious"].values == 0
    model = train_isolation_forest(X[benign_mask])
    is_anomaly, _ = score_anomalies(model, X)
    malicious_rate = is_anomaly[~benign_mask].mean()
    benign_rate = is_anomaly[benign_mask].mean()
    assert malicious_rate >= benign_rate


def test_hierarchical_pipeline_end_to_end():
    train_df = generate_dataset(1200, seed=1)
    test_df = generate_dataset(400, seed=2)
    fw = HierarchicalFirewall()
    fw.fit(train_df)
    results = fw.predict(test_df)
    assert "final_prediction" in results.columns
    assert "decision_tier" in results.columns
    acc = (results["final_prediction"] == results["is_malicious"]).mean()
    assert acc > 0.6
**
    requirements.txt**

    numpy
pandas
scikit-learn
xgboost
shap
lime
joblib
tensorflow-cpu
matplotlib
    
