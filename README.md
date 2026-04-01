# 🌸 Iris Species Classifier — Streamlit App

A machine learning web application that classifies Iris flowers into three species
(**setosa**, **versicolor**, **virginica**) using a **Decision Tree Classifier**,
deployed via **Streamlit Community Cloud**.

🔗 **Live App:** `https://your-app-name.streamlit.app` *(update after deployment)*

---

## Features

| Tab | What you get |
|-----|-------------|
| 🏠 Overview | Dataset summary, KPI cards, species distribution pie chart, data preview |
| 📊 Explore Data | Box plots, interactive scatter, correlation heatmap, filter by species |
| 🎯 Model Performance | Confusion matrix, feature importance, decision boundary chart |
| 🔮 Live Predict | Sidebar sliders → real-time prediction + confidence bars + scatter overlay |

---

## Project Structure

```
iris-streamlit/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── data/
│   └── iris.csv                    # Training dataset (150 samples)
├── metadata/
│   └── iris_classifier_v1.json     # Model metadata & metrics
├── models/
│   ├── iris_classifier_model_v1.joblib
│   └── iris_classifier_features_v1.joblib
├── src/
│   ├── app.py
│   ├── data_processor.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── model_registry.py
│   ├── training.py
│   └── config/
│       ├── __init__.py
│       └── config.ini
└── README.md
```

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (run once, artifacts already included)
python -m src.training --data_path data/iris.csv

# Launch app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Deploy to Streamlit Community Cloud

1. Push this repo to **GitHub** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — done in ~2 minutes!

---

## ML Details

| Item | Value |
|------|-------|
| Algorithm | Decision Tree Classifier |
| Dataset | Iris (sklearn built-in) |
| Features | sepal_length, sepal_width, petal_length, petal_width |
| Target | species (0=setosa, 1=versicolor, 2=virginica) |
| Train/Test | 80% / 20% stratified |
| Max Depth | 4 |
| Accuracy | 93.3% |
