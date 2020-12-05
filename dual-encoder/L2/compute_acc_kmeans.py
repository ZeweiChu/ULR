import numpy as np
import sys
from sklearn.metrics import f1_score
from sklearn.kernel_approximation import RBFSampler 
import code

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

text_embeddings = np.loadtxt(text_file)
cat_embeddings = np.loadtxt(cat_file)
num_cats = cat_embeddings.shape[0]

def euclidean(a, b):
    return np.sum((a - b) * (a-b), -1)

centroids = cat_embeddings
all_accs = []
all_scores = []

for k in range(100):
    centroids = np.expand_dims(centroids, 0)
    text_embeddings_unsqueezed = np.expand_dims(text_embeddings, 1)
    scores = euclidean(text_embeddings_unsqueezed, centroids)
    avg_score = np.mean(scores.min(1))
    all_scores.append(avg_score)

    new_preds = scores.argmin(1) 
    if k > 0:
        change_of_preds = np.sum(preds != new_preds)
        print("change of preds: ", change_of_preds)
        print("average score: ", avg_score)

    # code.interact(local=locals())

    preds = new_preds
    acc = np.sum(preds == labels) / len(labels)
    all_accs.append(acc)
    
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

min_index = np.argmin(all_scores)
print("min avg score: ", all_scores[min_index], ", the corresponding accuracy: ", all_accs[min_index], ", max accuracy: ", max(all_accs))
