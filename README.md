# ECG Arrhythmia Edge Classifier

**Course:** MSML612 Deep Learning
**Project:** Edge-Optimized Hybrid CNN-Transformer for Arrhythmia Detection

A lightweight hybrid 1D-CNN + Transformer model that classifies 3-second single-lead ECG windows as **Normal** or **Abnormal**, trained on the MIT-BIH Arrhythmia Database and quantized to INT8 for deployment on an **Arduino Nano 33 BLE Sense**.

---

## What's in this repo

| File | Purpose |
|---|---|
| `InferenceOnly.ipynb` | Notebook created for inference without re-training. 
| `model_builder.py` | Defines the model architecture: CNN feature extractor → relative positional encoding → Transformer encoder (×2) → classification head. Supports both a standard float32 build and a Quantization-Aware Training (QAT) build. |
| `generate_data.py` | Downloads/reads MIT-BIH records, segments them into 3-second (1,080-sample) windows, labels them, and applies the patient-grouped train/test split. Produces the `.npy` arrays used for training (not included in this repo — see below). |
| `training_pipeline_v2.ipynb` | Full training + evaluation notebook: trains the standard and QAT models, calibrates the decision threshold on validation data, evaluates on the held-out test set, converts the QAT model to INT8 TFLite, and exports it. |
|`training_pipeline_secondaryrun.ipynb`| Notebook containing results of a secondary run. The results obtained are presented in the report, but do not match version loaded onto Arduino|
| `model.tflite` | Final INT8-quantized model, ready to run with TFLite Micro. |
| `model_data.h` | The same model exported as a C byte array, ready to compile directly into Arduino firmware. |
| `qat_model` | Saved weights for trained QAT model. Loaded by InferenceOnly Notebook|
| `original_model` | Saved model without quantization nodes. Loaded by InferenceOnly Notebook|
|`ArduinoBoardCode.ino` |This is the code necessary for model to run on the Arduino board. It needs to be placed in the same folder as the 'model_data.h' file and uploaded to a board using ArduinoIDE.|
|`HardwareinLoopCode.ipynb` | This is the Python code used to communicate with the Arduino board. The COM port will likely need changed based on system|
|`Arduino_TensorFlowLite.zip` | This is the necessary Arduino libraries for ArduinoBoardCode. Drag this file into an existing Arduino library file. Contains backpatched Batch_MatMul changes|

**Not included** (see `.gitignore`): raw MIT-BIH data (`data/`), processed `.npy` arrays (`processed/`), and superseded drafts (`data 3.py`, `model Binary3.ipynb`, `model_v2.ipynb`, `Training-Pipeline-Binary.ipynb`) — `model_builder.py` and `training_pipeline_v2.ipynb` are the current, most-recent versions of those.

---

## Architecture summary

```
Raw ECG window (1080, 1)
   -> 1D-CNN feature extractor        (3 conv blocks, 1080 -> 360 -> 180 -> 90 timesteps)
   -> Relative positional encoding    (depthwise conv, residual)
   -> Transformer encoder block x2    (multi-head self-attention + feed-forward)
   -> Classification head             (global average pool -> dropout -> softmax)
-> Normal vs. Abnormal
```

~22,938 parameters (float32, ~89.6 KB) → ~69 KB after INT8 quantization.

## Evaluation methodology

- **Patient-grouped split** — training/validation/test partitions are split by patient ID (not by individual window) so no patient's data leaks across sets.
- **Macro F1** as the primary metric — averages F1 across Normal and Abnormal equally, since Abnormal is the minority class.
- **Threshold calibration** — the classification threshold is tuned on validation data only, separately for the float32, QAT, and final INT8 models.
- **Test set touched exactly once** — after the split, metric, and threshold are all finalized.

Test set results: macro F1 ≈ 0.70, accuracy ≈ 75–76% across float32, QAT, and INT8 variants.

---

## Reproducing this project locally

### 1. Clone the repo
```bash
git clone https://github.com/ayekhan12/ecg-arrhythmia-edge-classifier.git
cd ecg-arrhythmia-edge-classifier
```

### 2. Install dependencies
```bash
pip install tensorflow tf-keras tensorflow-model-optimization scikit-learn numpy matplotlib seaborn wfdb pyserial
```

### 3. Get the raw MIT-BIH data
This repo does not include the raw ECG records. Download the MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/) and place the `.dat`/`.atr`/`.hea` files in a local `data/` folder, or let `generate_data.py` fetch them automatically if it has that capability.

### 4. Generate the processed dataset
```bash
python generate_data.py
```
This creates a local `processed/` folder containing `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `patient_train.npy`, `patient_test.npy`.

### 5. Run the training pipeline
Open `training_pipeline_v2.ipynb` in Jupyter and run all cells. It imports `model_builder.py` directly, trains the standard and QAT models, evaluates on the test set, and re-exports `model.tflite` / `model_data.h`.

### 6. Check Inference
Open `InferenceOnly.ipynb` in Jupyter and run all cells. It imports the saved models, `model_builder.py` to rebuild the QAT model, and evaluates the models on the test set.

### 7. Flash the board
Copy `model_data.h` and place in same folder location as `ArduinoBoardCode.ino`. Uncompress `Arduino_TensorFlowLite.zip` into your Arduino library file. Flash Arduino code and model to board using ArduinoIDE.

### 8. Test On-board Deployment
Run `HardwareinLoopCode.ipynb`. Update serialport variable to match COM PORT as needed. 

---

## Team

- Ayesha Khan, Sriram Vema, Thaddeus Waterman

## Deployment target

Arduino Nano 33 BLE Sense (256 KB SRAM / 1 MB Flash) via TFLite Micro. `model_data.h` compiles directly into firmware.
