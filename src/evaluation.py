"""
This module contains evaluation procedures for the trained model.
"""

import logging
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)
from src.config import appconfig
from src import model_registry

logging.basicConfig(level=logging.INFO)

accuracy_min = float(appconfig['Evaluation']['accuracy'])
model_name = appconfig['Model']['name']
SPECIES = ["setosa", "versicolor", "virginica"]


def get_eval_metrics(y_true, y_pred):
    """Return a dict of evaluation metrics."""
    return {
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, average='macro'), 4),
        'recall':    round(recall_score(y_true, y_pred, average='macro'), 4),
        'f1':        round(f1_score(y_true, y_pred, average='macro'), 4),
    }


def run(y_true, y_pred):
    """Evaluate model; return True if it passes the quality gate."""
    logging.info("Evaluating model...")
    new_acc = round(accuracy_score(y_true, y_pred), 4)
    logging.info(f"\n{classification_report(y_true, y_pred, target_names=SPECIES)}")

    if new_acc < accuracy_min:
        logging.warning(f"Failed: accuracy {new_acc} below minimum {accuracy_min}")
        return False

    current = model_registry.get_metadata(model_name)
    if current:
        current_acc = current.get('metrics', {}).get('accuracy', 0)
        if new_acc < current_acc:
            logging.warning(f"Failed: accuracy {new_acc} does not beat current {current_acc}")
            return False

    logging.info(f"Evaluation passed — accuracy: {new_acc}")
    return True
