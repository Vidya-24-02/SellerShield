# 🛡️ SellerShield — Combined Real ML + Beautiful UI

This combines the exact original HTML design with a real trained ML model backend.

---

## 📁 Project Structure
```
SellerShield_Combined/
├── index.html          ← Beautiful UI (exact same design, now calls real ML)
├── server.py           ← Flask backend serving the ML model via API
├── train_model.py      ← Trains the real ML model (run once)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── model/              ← Created after training
    ├── seller_model.pkl
    ├── feature_names.pkl
    └── classes.pkl
```

---

## 🚀 Setup — 3 Steps

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the ML model (ONE TIME, ~60-90 seconds)
```bash
python train_model.py
```

### Step 3 — Start the server
```bash
python server.py
```

**Open browser → http://localhost:5000**

---

## ✅ What's Real ML Here

| Component | Detail |
|---|---|
| Algorithm | VotingClassifier (RandomForest 200 trees + GradientBoosting 150 trees) |
| Training | 50,000 synthetic seller profiles |
| Accuracy | ~82% test accuracy, 5-fold cross-validated |
| API | Flask `/api/predict` receives seller data → returns real prediction |
| Features | 20 features fed into StandardScaler → trained model |

The HTML sends your slider values to `/api/predict` via `fetch()`.
Flask runs them through the REAL trained `.pkl` model and returns the score.

---

## How it Works (Flow)
```
Browser (index.html)
    ↓ fetch POST /api/predict  (seller metrics as JSON)
Flask server.py
    ↓ builds feature DataFrame
    ↓ StandardScaler normalizes
    ↓ RandomForest + GradientBoosting predict
    ↓ returns { score, label, probabilities, explanation, flags }
Browser
    ↓ renders beautiful results UI
```

© 2025 SellerShield
