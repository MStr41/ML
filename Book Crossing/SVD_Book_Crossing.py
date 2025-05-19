# -*- coding: utf-8 -*-
"""SVD_RecPack_BookCrossing_UserBasedSplit+10_Core_Pruning.ipynb"""

import recpack.pipelines as pipelines
from recpack.scenarios import WeakGeneralization
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, Deduplicate
import numpy as np
import pandas as pd
import joblib
import json
from recpack.preprocessing.preprocessors import DataFramePreprocessor

# Set random seed for reproducibility
np.random.seed(42)

# === LOAD AND PREPROCESS BOOK-CROSSING DATASET ===

# Load Book-Crossing ratings file (downloaded from Kaggle and extracted)
file_path = 'BX-Book-Ratings.csv'
ratings = pd.read_csv(file_path, sep=';', encoding='latin-1')

# Rename columns for consistency
ratings.columns = ['user_id', 'item_id', 'rating']

# Filter only explicit ratings (rating > 0)
ratings = ratings[ratings['rating'] > 0]

# Convert user and item IDs to integers
ratings['user_id'], user_index = pd.factorize(ratings['user_id'])
ratings['item_id'], item_index = pd.factorize(ratings['item_id'])

# Ensure ratings are floats
ratings['rating'] = ratings['rating'].astype(float)

# Remove duplicate ratings (same user, same item): average them
ratings = ratings.groupby(['user_id', 'item_id'], as_index=False)['rating'].mean()

# === 10-CORE PRUNING FUNCTION ===
def prune_10_core(data):
    while True:
        user_counts = data['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 10].index
        data = data[data['user_id'].isin(valid_users)]

        item_counts = data['item_id'].value_counts()
        valid_items = item_counts[item_counts >= 10].index
        data = data[data['item_id'].isin(valid_items)]

        if all(user_counts >= 10) and all(item_counts >= 10):
            break
    return data

# Apply 10-core pruning
ratings = prune_10_core(ratings)

# === RecPack-PREPROCESSING ===
proc = DataFramePreprocessor(item_ix='item_id', user_ix='user_id')
interaction_matrix = proc.process(ratings)

# Sanity checks
interaction_matrix = interaction_matrix.users_in(interaction_matrix.active_users)
interaction_matrix = interaction_matrix.items_in(interaction_matrix.active_items)

# === SPLITTING: Weak Generalization + Validation/Test Split ===
test_valid_fraction = 0.229
weak_gen_scenario = WeakGeneralization(frac_data_in=1 - test_valid_fraction, validation=False, seed=42)
weak_gen_scenario.split(interaction_matrix)

train_interactions = weak_gen_scenario.full_training_data
test_valid_interactions = weak_gen_scenario.test_data_out

# Further split test_valid into validation and test sets
test_valid_scenario = WeakGeneralization(frac_data_in=0.40, validation=False, seed=42)
test_valid_scenario.split(test_valid_interactions)

valid_interactions = test_valid_scenario.full_training_data
test_out_interactions = test_valid_scenario.test_data_out

# === DOWNSAMPLING TRAINING DATA ===
downsample_fraction = 1.0
additional_split_scenario = WeakGeneralization(frac_data_in=downsample_fraction, validation=False, seed=42)
additional_split_scenario.split(train_interactions)
downsampled_train_interactions = additional_split_scenario.full_training_data

# === BUILD RecPack PIPELINE ===
pipeline_builder = pipelines.PipelineBuilder('BookCrossing_SVD')

pipeline_builder.set_full_training_data(downsampled_train_interactions)
pipeline_builder.set_test_data((downsampled_train_interactions, test_out_interactions))
pipeline_builder.set_validation_training_data(downsampled_train_interactions)
pipeline_builder.set_validation_data((downsampled_train_interactions, valid_interactions))

# Add SVD algorithm with grid search over latent dimensions
pipeline_builder.add_algorithm(
    'SVD',
    grid={
        'num_components': [20, 30, 60, 80, 100, 200],
        'seed': [42]
    }
)

pipeline_builder.set_optimisation_metric('NDCGK', K=10)
pipeline_builder.add_metric('NDCGK', [10])

pipeline = pipeline_builder.build()
pipeline.run()

# === RESULTS ===
print("Metric Results:")
print(pipeline.get_metrics())

print("Best Hyperparameters:")
print(pipeline.optimisation_results)
