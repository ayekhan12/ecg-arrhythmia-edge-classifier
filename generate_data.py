"""
MIT-BIH Arrhythmia Database — Data Preparation

Downloads the MIT-BIH Arrhythmia Database (if not already present), segments
each record into 3-second sliding windows (1,080 samples @ 360 Hz, 1-second
stride), assigns each window an AAMI class label (worst-case priority: a
window containing any Ventricular beat is labeled Ventricular even if most
of its beats are Normal), and writes the result to `processed/`.

Uses the standard inter-patient train/test record split (de Chazal et al.,
2004) so that no patient's beats ever appear in both the training and test
sets.

Run this once before `Training-Pipeline-Binary.ipynb`. It skips the download
step if `data/mitdb` already contains record 100.
"""

import os
import numpy as np
import wfdb
from tqdm import tqdm

# ── Signal / windowing configuration ───────────────────────────────────────
FS = 360                               # MIT-BIH sampling frequency (Hz)

WINDOW_SECONDS = 3
WINDOW_SIZE = FS * WINDOW_SECONDS      # 1,080 samples per window

STEP_SECONDS = 1
STEP_SIZE = FS * STEP_SECONDS          # 360-sample stride (overlapping windows)

DATA_DIR = "data/mitdb"

# Inter-patient train/test split (de Chazal et al., 2004).
TRAIN_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",
]

TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
]

# ── AAMI class mapping ──────────────────────────────────────────────────────
AAMI_MAP = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown
    "/": "Q", "f": "Q", "Q": "Q",
}

# Worst-case labeling: if a window contains beats from multiple classes,
# prioritize the most clinically significant one.
LABEL_PRIORITY = ["V", "S", "F", "Q", "N"]

LABEL_TO_INT = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]


def normalize(signal):
    """Zero-mean, unit-variance normalization of the raw ECG signal."""
    return (signal - np.mean(signal)) / np.std(signal)


def get_window_label(labels):
    """Resolve a window's beat labels to a single AAMI class by priority."""
    labels = set(labels)
    for label in LABEL_PRIORITY:
        if label in labels:
            return label
    return None


def process_record(record_name):
    """
    Segment one MIT-BIH record into overlapping 3-second windows and assign
    each window an AAMI class label.

    Returns:
        X: (num_windows, WINDOW_SIZE) array of normalized signal windows
        y: (num_windows,) array of integer class labels
        patient_ids: (num_windows,) array of the source record name
    """
    record = wfdb.rdrecord(os.path.join("data", record_name))
    ann = wfdb.rdann(os.path.join("data", record_name), "atr")

    signal = normalize(record.p_signal[:, 0])
    beat_samples = ann.sample
    beat_symbols = ann.symbol

    X, y, patient_ids = [], [], []

    for start in range(0, len(signal) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        labels = []
        for sample, symbol in zip(beat_samples, beat_symbols):
            if sample < start:
                continue
            if sample >= end:
                break
            if symbol in AAMI_MAP:
                labels.append(AAMI_MAP[symbol])

        if len(labels) == 0:
            continue

        label = get_window_label(labels)
        if label is None:
            continue

        X.append(signal[start:end])
        y.append(LABEL_TO_INT[label])
        patient_ids.append(record_name)

    return np.array(X), np.array(y), np.array(patient_ids)


def build_dataset(records):
    """Process a list of records and concatenate their windows."""
    X_all, y_all, patient_all = [], [], []

    for record in tqdm(records):
        X, y, patients = process_record(record)
        X_all.extend(X)
        y_all.extend(y)
        patient_all.extend(patients)

    return np.array(X_all), np.array(y_all), np.array(patient_all)


def main():
    os.makedirs("data", exist_ok=True)
    records = TRAIN_RECORDS + TEST_RECORDS

    # Only download records that aren't already on disk.
    missing = [r for r in records if not os.path.exists(os.path.join("data", f"{r}.hea"))]
    if missing:
        print(f"Downloading {len(missing)} missing MIT-BIH record(s): {missing}")
        for record in missing:
            wfdb.dl_database("mitdb", dl_dir="data", records=[record])
        print("Download complete.")
    else:
        print("MIT-BIH dataset already exists. Skipping download.")

    print("\nBuilding training set...")
    X_train, y_train, patient_train = build_dataset(TRAIN_RECORDS)

    print("\nBuilding test set...")
    X_test, y_test, patient_test = build_dataset(TEST_RECORDS)

    print(f"\nTraining set: X={X_train.shape} y={y_train.shape}")
    print(f"Test set:     X={X_test.shape} y={y_test.shape}")

    os.makedirs("processed", exist_ok=True)
    np.save("processed/X_train.npy", X_train)
    np.save("processed/y_train.npy", y_train)
    np.save("processed/X_test.npy", X_test)
    np.save("processed/y_test.npy", y_test)
    np.save("processed/patient_train.npy", patient_train)
    np.save("processed/patient_test.npy", patient_test)
    print("\nSaved to processed/")

    print("\nTraining set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {CLASS_NAMES[u]}: {c:,}")


if __name__ == "__main__":
    main()
