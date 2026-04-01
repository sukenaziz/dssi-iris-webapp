"""
This module manages persistence and versioning of trained models.
"""

import json
import os
from joblib import dump, load
from datetime import datetime
from src.config import appconfig
import logging

logging.basicConfig(level=logging.INFO)

MODEL_DIR = appconfig['Directories']['models']
METADATA_DIR = appconfig['Directories']['metadata']


def get_next_version(model_name):
    """Determine the next version number for a given model."""
    versions = [0]
    for file in os.listdir(METADATA_DIR):
        if file.startswith(model_name):
            try:
                version = int(file.split('_v')[-1].split('.')[0])
                versions.append(version)
            except ValueError:
                pass
    return max(versions) + 1


def register(model, features, metadata):
    """Register a new model and its metadata."""
    version = get_next_version(metadata['name'])
    model_file_name = f"{metadata['name']}_model_v{version}.joblib"
    features_file_name = f"{metadata['name']}_features_v{version}.joblib"

    metadata['version'] = version
    metadata['registration_date'] = datetime.now().isoformat()

    dump(model, os.path.join(MODEL_DIR, model_file_name))
    metadata['model'] = model_file_name

    dump(features, os.path.join(MODEL_DIR, features_file_name))
    metadata['features'] = features_file_name

    metadata_file_name = f"{metadata['name']}_v{version}.json"
    metadata_path = os.path.join(METADATA_DIR, metadata_file_name)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Registered: {metadata['name']} v{version}")
    return metadata_path


def get_metadata(model_name, version=None):
    """Retrieve metadata. Returns None if not found."""
    if version is None:
        version = get_next_version(model_name) - 1
    path = os.path.join(METADATA_DIR, f"{model_name}_v{version}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def retrieve(model_name, version=None):
    """Load and return model + features list."""
    if version is None:
        version = get_next_version(model_name) - 1
    path = os.path.join(METADATA_DIR, f"{model_name}_v{version}.json")
    with open(path, 'r') as f:
        metadata = json.load(f)
    model = load(os.path.join(MODEL_DIR, metadata["model"]))
    features = load(os.path.join(MODEL_DIR, metadata["features"]))
    return model, features
