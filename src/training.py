"""
Training pipeline for Iris Species Classifier.
Algorithm : Decision Tree Classifier
Dataset   : Iris (150 samples, 4 features, 3 classes)
"""

import argparse
import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src import data_processor, model_registry, evaluation
from src.config import appconfig

logging.basicConfig(level=logging.INFO)

features = appconfig['Model']['features'].split(',')
label    = appconfig['Model']['label']


def run(data_path):
    logging.info("Loading data...")
    df = data_processor.run(data_path)

    logging.info("Train-Test Split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[label],
        test_size=appconfig.getfloat('Model', 'test_size'),
        random_state=42,
        stratify=df[label]
    )

    logging.info("Training Decision Tree...")
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if evaluation.run(y_test, y_pred):
        logging.info("Persisting model...")
        mdl_meta = {
            'name': appconfig['Model']['name'],
            'algorithm': 'DecisionTreeClassifier',
            'metrics': evaluation.get_eval_metrics(y_test, y_pred)
        }
        model_registry.register(clf, features, mdl_meta)

    logging.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    run(args.data_path)
