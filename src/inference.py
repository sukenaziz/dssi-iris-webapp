"""
This module encapsulates model inference.
"""

import pandas as pd
from src.data_processor import preprocess
from src.model_registry import retrieve
from src.config import appconfig

SPECIES_MAP  = {0: "setosa", 1: "versicolor", 2: "virginica"}
SPECIES_EMOJI = {0: "🌸", 1: "🌿", 2: "🌺"}


def get_prediction(**kwargs):
    """
    Run inference and return species id, name, emoji, and probabilities.
        Parameters:
            kwargs: sepal_length, sepal_width, petal_length, petal_width
        Returns:
            dict: prediction results
    """
    clf, features = retrieve(appconfig['Model']['name'])
    df = pd.DataFrame([kwargs])
    df = preprocess(df)
    pred_id   = int(clf.predict(df[features])[0])
    proba     = clf.predict_proba(df[features])[0]
    return {
        "species_id":   pred_id,
        "species_name": SPECIES_MAP[pred_id],
        "emoji":        SPECIES_EMOJI[pred_id],
        "probabilities": {
            SPECIES_MAP[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
    }
