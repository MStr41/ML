import pandas as pd
import numpy as np
from lenskit.algorithms import item_knn as knn
from lenskit import batch, util, crossfold as xf, Recommender
import seedbank
import json
import sys

# -------------------------------
# 1. LOAD & CLEAN THE DATA
# -------------------------------

# Load the Book-Ratings file
ratings = pd.read_csv('Book_Crossing_Dataset\BX-Book-Ratings.csv', sep=';', encoding='latin-1',
                      usecols=['User-ID', 'ISBN', 'Book-Rating'])

# Rename columns
ratings.columns = ['user', 'item', 'rating']

# Remove interactions with 0 rating (implicit or unknown)
ratings = ratings[ratings['rating'] > 0]

# Factorize user and item IDs into integer indices
ratings['user'], user_index = pd.factorize(ratings['user'])
ratings['item'], item_index = pd.factorize(ratings['item'])

# -------------------------------
# 2. 10-CORE PRUNING
# -------------------------------

def prune_10_core(data):
    while True:
        user_counts = data['user'].value_counts()
        item_counts = data['item'].value_counts()

        valid_users = user_counts[user_counts >= 10].index
        valid_items = item_counts[item_counts >= 10].index

        data = data[data['user'].isin(valid_users)]
        data = data[data['item'].isin(valid_items)]

        if data['user'].value_counts().min() >= 10 and data['item'].value_counts().min() >= 10:
            break
    return data

ratings = prune_10_core(ratings)

print("Users:", ratings['user'].nunique())
print("Items:", ratings['item'].nunique())
print("Ratings:", len(ratings))

# -------------------------------
# 3. SPLIT THE DATA (Train-Test-Validation)
# -------------------------------

# 10% as test set
final_test_method = xf.SampleFrac(0.10, rng_spec=42)
train_parts, test_parts = [], []

for part in xf.partition_users(ratings, 1, final_test_method):
    train_parts.append(part.train)
    test_parts.append(part.test)

train_data = pd.concat(train_parts)
final_test_data = pd.concat(test_parts)

# 11.11% as validation set from training data
validation_split_method = xf.SampleFrac(0.1111, rng_spec=42)
train_parts, valid_parts = [], []

for part in xf.partition_users(train_data, 1, validation_split_method):
    train_parts.append(part.train)
    valid_parts.append(part.test)

pure_train_data = pd.concat(train_parts)
validation_data = pd.concat(valid_parts)

# Downsample the training data (use 50%)
##########################################################################
try:
    fraction_value = float(sys.argv[1])  
except (IndexError, ValueError):
    fraction_value = 1.0
downsample_fraction = fraction_value
##########################################################################
downsample_method = xf.SampleFrac(1.0 - fraction_value, rng_spec=42)
downsampled_parts = []

for part in xf.partition_users(pure_train_data, 1, downsample_method):
    downsampled_parts.append(part.train)

downsampled_train_data = pd.concat(downsampled_parts)

# -------------------------------
# 4. NDCG CALCULATION CLASS
# -------------------------------

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
            relevance = 1 if item in self.test_items else 0
            rank = i + 1
            dcg += relevance / np.log2(rank + 1)
        return dcg

    def calculate(self):
        dcg = self.calculate_dcg()
        ideal_dcg = self._ideal_dcg()
        return dcg / ideal_dcg if ideal_dcg > 0 else 0

# -------------------------------
# 5. EVALUATION FUNCTION (nDCG@10)
# -------------------------------

def evaluate_with_ndcg(aname, algo, train, valid):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = valid['user'].unique()
    recs = batch.recommend(fittable, users, 10, n_jobs=1)
    recs['Algorithm'] = aname

    total_ndcg = 0
    for user in users:
        user_recs = recs[recs['user'] == user]['item'].values
        user_truth = valid[valid['user'] == user]['item'].values
        total_ndcg += nDCG_LK(10, user_recs, user_truth).calculate()

    mean_ndcg = total_ndcg / len(users)
    return recs, mean_ndcg

# -------------------------------
# 6. TRY DIFFERENT K VALUES
# -------------------------------

results = []
best_k = None
best_mean_ndcg = -float('inf')
k_values = [5, 10, 15, 20, 30, 40, 60, 80, 100]

for k in k_values:
    seedbank.initialize(42)
    algo_ii = knn.ItemItem(nnbrs=k, center=False, aggregate='sum', feedback="explicit")
    _, mean_ndcg = evaluate_with_ndcg('ItemItem', algo_ii, downsampled_train_data, validation_data)
    results.append({'K': k, 'Mean nDCG': mean_ndcg})

    if mean_ndcg > best_mean_ndcg:
        best_mean_ndcg = mean_ndcg
        best_k = k

print("\nBest K value and results:")
for result in results:
    print(f"K = {result['K']}: Mean nDCG = {result['Mean nDCG']:.4f}")
print(f"\nBest K = {best_k} (nDCG = {best_mean_ndcg:.4f})")

# -------------------------------
# 7. FINAL EVALUATION ON TEST SET
# -------------------------------

final_algo = knn.ItemItem(nnbrs=best_k, center=False, aggregate='sum', feedback="explicit")
_, final_test_ndcg = evaluate_with_ndcg('ItemItem', final_algo, downsampled_train_data, final_test_data)

print(f"\n Final Test Set nDCG@10: {final_test_ndcg:.4f}")

#################################################
ndcg_value = final_test_ndcg
key_name = "item_knn_book_crossing"

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
