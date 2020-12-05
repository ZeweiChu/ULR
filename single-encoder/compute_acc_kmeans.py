import numpy as np
import sys
from sklearn.metrics import f1_score
from scipy.spatial import distance
from scipy.special import softmax, kl_div
import code

if len(sys.argv) != 3:
    print("Usage: python compute_acc.py preds_file test_file")
    exit(-1)

test_file = sys.argv[1]
preds_file = sys.argv[2]

with open(test_file) as fin:
    labels = []
    for line in fin:
        label, _ = line.split(",", 1)
        if label.startswith("'") or label.startswith('"'):
            label = label[1:-1]
        labels.append(int(label) - 1)

    labels = np.array(labels)

preds = np.loadtxt(preds_file)
num_cats = preds.shape[1]
num_ins = preds.shape[0]
probs = softmax(preds, axis=1)
preds = preds.argmax(1) 

acc = np.sum(preds == labels) / len(labels)

print("accuracy of {}: {}".format(preds_file, acc))

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0), -1)


centroids = np.zeros((num_cats, num_cats))
cat_embeddings = np.zeros((num_cats, num_cats))
for i in range(num_cats):
    centroids[i][i] = 1
    cat_embeddings[i][i] = 1

all_avg_scores = []
all_accs = []
for k in range(100):
    old_preds = preds.copy()
    scores = np.zeros((num_ins, num_cats))
    
    
    for i in range(num_ins):
        for j in range(num_cats):
            scores[i, j] = distance.jensenshannon(probs[i], centroids[j])

    preds = scores.argmin(1)

    avg_score = np.mean(scores.min(1))
    all_avg_scores.append(avg_score)

    acc = np.sum(preds == labels) / len(labels)
    all_accs.append(acc)
    num_updates = np.sum(preds != old_preds)
    print("iteration: ", k, ", accuracy: ", acc, ", number of updates: ", num_updates)
    if num_updates == 0 and k > 0:
        break
    new_centroids = []
    for j in range(num_cats):
        new_centroid = probs[preds == j]
        if len(new_centroid) == 0:
            new_centroid = centroids[j][None, :]
        else:
            new_centroid = new_centroid.mean(0, keepdims=True)
        new_centroids.append(new_centroid)
    centroids = np.concatenate(new_centroids, axis=0)
