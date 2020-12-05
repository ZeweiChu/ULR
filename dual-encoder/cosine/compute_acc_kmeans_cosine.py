import numpy as np
import sys
from sklearn.metrics import f1_score
import code
import random


random.seed(1)

if len(sys.argv) not in [4]:
    print("Usage: python compute_acc.py test_file text_embeddings category_embeddings ")
    exit(-1)

test_file = sys.argv[1]
text_file = sys.argv[2]
cat_file = sys.argv[3]

with open(test_file) as fin:
    labels = []
    for line in fin:
        label, _ = line.split(",", 1)
        if label.startswith("'") or label.startswith('"'):
            label = label[1:-1]
        labels.append(int(label) - 1)

    labels = np.array(labels)

def normalize(x):
    return x / np.sum(x * x, -1, keepdims=True)


text_embeddings = np.loadtxt(text_file)
cat_embeddings = np.loadtxt(cat_file)
num_cats = cat_embeddings.shape[0]

text_embeddings = normalize(text_embeddings)
cat_embeddings = normalize(cat_embeddings)


def dot_product(a, b):
    return np.sum(a * b, -1) 

centroids = cat_embeddings

all_avg_scores = []
all_accs = []

for k in range(100):
    centroids = np.expand_dims(centroids, 0)
    text_embeddings_unsqueezed = np.expand_dims(text_embeddings, 1)

    scores = dot_product(text_embeddings_unsqueezed, centroids)
    avg_score = np.mean(scores.max(1))
    all_avg_scores.append(avg_score)
    new_preds = scores.argmax(1) 

    if k > 0:
        change_of_preds = np.sum(preds != new_preds)
        print("change of preds: ", change_of_preds)


    preds = new_preds
    acc = np.sum(preds == labels) / len(labels)
    all_accs.append(acc)
    
    print("average score: ", avg_score)
    print("after {} iterations of k means accuracy: {}".format(k, acc))

    centroids = []
    for i in range(num_cats):
        centroid = text_embeddings[preds == i]
        if centroid.shape[0] > 0:
            centroids.append(centroid.mean(0, keepdims=True))
        else:
            centroids.append(text_embeddings.mean(0, keepdims=True))
        
    centroids = np.concatenate(centroids, 0)
    centroids = (centroids + cat_embeddings) / 2
    
    if k > 0 and change_of_preds == 0:
        break


max_index = np.argmax(all_avg_scores)
print("max avg score: ", all_avg_scores[max_index], ", the corresponding accuracy: ", all_accs[max_index], ", max accuracy: ", max(all_accs))
