import csv
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

CSV_IN   = "piano_notes_train.csv" 
MODEL_OUT = "knn_model.pkl"

# charge X, y ---------------------------------------------------------------
X, y = [], []
with open(CSV_IN, newline="") as f:
    next(csv.reader(f), None)               # ← sauter la ligne d'en-tête
    for row in csv.reader(f):
        *feat_s, label_s = row    # ← séparer features et label
        X.append([float(v) for v in feat_s])
        y.append(int(label_s))

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int16)

# pipeline : normalisation + KNN -------------------------------------------
pipe = make_pipeline(
    Normalizer(norm="l2"),
    KNeighborsClassifier(n_neighbors=1, metric="euclidean")
)
pipe.fit(X, y)

joblib.dump(pipe, MODEL_OUT)
print(" Modèle sauvegardé :", MODEL_OUT)
