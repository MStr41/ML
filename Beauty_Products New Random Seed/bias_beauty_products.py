# -*- coding: utf-8 -*-
from lenskit.algorithms import Recommender
from lenskit.algorithms.bias import Bias
from lenskit import batch, topn, util
import pandas as pd
import joblib
import gzip
import json
from lenskit import crossfold as xf
import seedbank
import numpy as np  

class nDCG_LK:
    def __init__(self, n, top_items, test_items):
        self.n = n
        self.top_items = top_items
        self.test_items = test_items

    def _ideal_dcg(self):
        iranks = np.zeros(self.n, dtype=np.float64)
        iranks[:] = np.arange(1, self.n + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=0)
        if len(self.test_items) < self.n:
            idcg[len(self.test_items):] = idcg[len(self.test_items) - 1]
        return idcg[self.n - 1]

    def calculate_dcg(self):
        dcg = 0
        for i, item in enumerate(self.top_items):
            if item in self.test_items:
                relevance = 1
            else:
                relevance = 0
            rank = i + 1
            contribution = relevance / np.log2(rank + 1)
            dcg += contribution
        return dcg

    def calculate(self):
        dcg = self.calculate_dcg()
        ideal_dcg = self._ideal_dcg()
        if ideal_dcg == 0:
            return 0
        ndcg = dcg / ideal_dcg
        return ndcg

seedbank.initialize(123)


# Load and preprocess ratings data
file_path = r'beauty_products_dataset/beauty_products_dataset.csv'
ratings = pd.read_csv(file_path, sep=',', encoding='latin-1',
                      usecols=['UserId', 'ProductId', 'Rating'])
ratings = ratings.rename(columns={'UserId': 'user', 'ProductId': 'item', 'Rating': 'rating'})
ratings = ratings.dropna(subset=['rating'])
# Convert 'rating' column to float
ratings['rating'] = ratings['rating'].astype(float)
# Keep only the necessary columns
ratings = ratings[['user', 'item', 'rating']]
print(ratings.head())

# Convert user and item IDs to integers
ratings['user'], user_index = pd.factorize(ratings['user'])
ratings['item'], item_index = pd.factorize(ratings['item'])
print(ratings.head())
print(len(ratings))

# Inspect the ratings data
print("Initial Ratings Data Inspection:")
print("Number of interactions:", len(ratings))
print("Number of unique users:", ratings['user'].nunique())
print("Number of unique items:", ratings['item'].nunique())

# Check for users and items with fewer than 10 interactions
user_counts = ratings['user'].value_counts()
item_counts = ratings['item'].value_counts()

print("\nUsers with fewer than 10 interactions:", (user_counts < 10).sum())
print("Items with fewer than 10 interactions:", (item_counts < 10).sum())

# Check for empty rows
empty_rows = ratings.isnull().sum().sum()
print("\nNumber of empty rows:", empty_rows)

# Check for duplicate rows
duplicate_rows = ratings.duplicated().sum()
print("Number of duplicate rows:", duplicate_rows)
# Check for duplicate ratings (same user, same item)
duplicate_ratings = ratings.duplicated(subset=['user', 'item']).sum()
print("Number of duplicate ratings (same user, same item):", duplicate_ratings)

# Remove duplicate rows
ratings = ratings.drop_duplicates()
# Aggregate duplicate ratings (same user, same item) by averaging their ratings
ratings = ratings.groupby(['user', 'item'], as_index=False)['rating'].mean()

# Check for empty rows after cleaning
empty_rows = ratings.isnull().sum().sum()
print("\nNumber of empty rows after cleaning:", empty_rows)

# Check for duplicate rows after cleaning
duplicate_rows = ratings.duplicated().sum()
print("Number of duplicate rows after cleaning:", duplicate_rows)

# Check for duplicate ratings (same user, same item) after cleaning
duplicate_ratings = ratings.duplicated(subset=['user', 'item']).sum()
print("Number of duplicate ratings (same user, same item) after cleaning:", duplicate_ratings)

# 10-core pruning
def prune_10_core(data):
    while True:
        # Filter users with fewer than 10 interactions
        user_counts = data['user'].value_counts()
        valid_users = user_counts[user_counts >= 10].index
        data = data[data['user'].isin(valid_users)]

        # Filter items with fewer than 10 interactions
        item_counts = data['item'].value_counts()
        valid_items = item_counts[item_counts >= 10].index
        data = data[data['item'].isin(valid_items)]

        # Check if no more pruning is needed
        if all(user_counts >= 10) and all(item_counts >= 10):
            break
    return data

# Apply 10-core pruning
ratings = prune_10_core(ratings)

# Inspect the pruned ratings data
print("\nAfter Pruning:")
print("Number of interactions:", len(ratings))
print("Number of unique users:", ratings['user'].nunique())
print("Number of unique items:", ratings['item'].nunique())

# Check for users and items with fewer than 10 interactions after pruning
user_counts = ratings['user'].value_counts()
item_counts = ratings['item'].value_counts()

print("\nUsers with fewer than 10 interactions after pruning:", (user_counts < 10).sum())
print("Items with fewer than 10 interactions after pruning:", (item_counts < 10).sum())

# Split into train and test sets
final_test_method = xf.SampleFrac(0.10, rng_spec=123)

train_parts = []
test_parts = []

for tp in xf.partition_users(ratings, 1, final_test_method):
    train_parts.append(tp.train)
    test_parts.append(tp.test)

train_data = pd.concat(train_parts)
final_test_data = pd.concat(test_parts)

# Check and print the number of interactions and users in each set

print("Train Data - Number of Interactions:", len(train_data))

print("Final Test Data - Number of Interactions:", len(final_test_data))

print("Train Data - Number of Users:", train_data['user'].nunique())

print("Final Test Data - Number of Users:", final_test_data['user'].nunique())

# Split train data into train and validation sets
validation_split_method = xf.SampleFrac(0.1111, rng_spec=123)

train_parts = []
valid_parts = []

for tp in xf.partition_users(train_data, 1, validation_split_method):
    train_parts.append(tp.train)
    valid_parts.append(tp.test)

pure_train_data = pd.concat(train_parts)
validation_data = pd.concat(valid_parts)

# Check and print the number of interactions and users in each set
print("\nBefore Splitting:")
print("Pure Train Data - Number of Interactions:", len(pure_train_data))
print("Validation Data - Number of Interactions:", len(validation_data))
print("Final Test Data - Number of Interactions:", len(final_test_data))

print("Pure Train Data - Number of Users:", pure_train_data['user'].nunique())
print("Validation Data - Number of Users:", validation_data['user'].nunique())
print("Final Test Data - Number of Users:", final_test_data['user'].nunique())


# Downsample the training set to different% of interactions for each user using xf.SampleFrac
##########################################################################
import sys
try:
    fraction_value = float(sys.argv[1])  
except (IndexError, ValueError):
    fraction_value = 0.7
downsample_fraction = fraction_value
##########################################################################
downsample_method = xf.SampleFrac(1.0 - downsample_fraction, rng_spec=123)
downsampled_train_parts = []

for i, tp in enumerate(xf.partition_users(pure_train_data, 1, downsample_method)):
    downsampled_train_parts.append(tp.train)

# Combine downsampled train parts into one DataFrame
downsampled_train_data = pd.concat(downsampled_train_parts)

# Checks for number of interactions and users in each set after downsampling
print("\nAfter Downsampling:")
print("Downsampled Train Data - Number of Interactions:", len(downsampled_train_data))
print("Validation Data - Number of Interactions:", len(validation_data))
print("Final Test Data - Number of Interactions:", len(final_test_data))

print("Downsampled Train Data - Number of Users:", downsampled_train_data['user'].nunique())
print("Validation Data - Number of Users:", validation_data['user'].nunique())
print("Final Test Data - Number of Users:", final_test_data['user'].nunique())

def evaluate_with_ndcg(aname, algo, train, valid):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = valid.user.unique()
    recs = batch.recommend(fittable, users, 10, n_jobs=1)
    recs['Algorithm'] = aname

    total_ndcg = 0
    for user in users:
        user_recs = recs[recs['user'] == user]['item'].values
        user_truth = valid[valid['user'] == user]['item'].values
        ndcg_score = nDCG_LK(10, user_recs, user_truth).calculate()
        total_ndcg += ndcg_score

    mean_ndcg = total_ndcg / len(users)
    return recs, mean_ndcg

# Perform validation and compute nDCG
algo_bias = Bias(damping = 1000)
valid_recs, mean_ndcg = evaluate_with_ndcg('Bias', algo_bias, downsampled_train_data, validation_data)
print(f"NDCG mean for validation set: {mean_ndcg:.4f}")

# Fit the algorithm on the full training data
final_algo = Bias(damping = 1000)

# Use evaluate_with_ndcg to get recommendations and mean nDCG
final_recs, mean_ndcg = evaluate_with_ndcg('Bias', final_algo, downsampled_train_data, final_test_data)
print(f"NDCG mean for test set: {mean_ndcg:.4f}")

#################################################
ndcg_value = mean_ndcg
key_name = "bias_beauty_products_newRandom"

from filelock import FileLock
import os
import json


output_file = "metric_results.json"
lock_file = output_file + ".lock"
fraction_key = str(downsample_fraction)

#Mit lock wird es gesichert
with FileLock(lock_file):
    # Datei lesen und schreiben
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                content = json.load(f)
                if not isinstance(content, dict):
                    content = {}
            except json.JSONDecodeError:
                content = {}
    else:
        content = {}

    if key_name not in content:
        content[key_name] = {}

    content[key_name][fraction_key] = ndcg_value

    with open(output_file, "w") as f:
        json.dump(content, f, indent=4)
#################################################