import csv
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CSV_IN   = "maps_train.csv"
MODEL_OUT = "knn_model.pkl"

# charge X, y ---------------------------------------------------------------
X, y = [], []
with open(CSV_IN) as f:
    for row in csv.reader(f):
        *feat, label = map(float, row)
        X.append(feat)
        y.append(int(label))

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int16)

# pipeline : normalisation + KNN -------------------------------------------
pipe = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, metric="euclidean")
)
pipe.fit(X, y)

joblib.dump(pipe, MODEL_OUT)
print(" Modèle sauvegardé :", MODEL_OUT)
