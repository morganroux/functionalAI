import json
from tslearn.clustering import TimeSeriesKMeans
import numpy as np

with open("points.json", "r", encoding="utf-8") as file:
    data = json.load(file)

for i, val in enumerate(data):
    new_item = []
    for key in val:
        new_item.append(val[key])
    data[i] = new_item

np_data = np.array(data)

# Génération de données de séries temporelles multidimensionnelles factices
# Chaque série temporelle a 3 dimensions et 5 points de temps
data = np.array(
    [
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
        [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
        [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
        [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
    ]
)

# Initialisation et ajustement du modèle KMeans pour les séries temporelles multidimensionnelles
model = TimeSeriesKMeans(n_clusters=2, metric="dtw")
labels = model.fit_predict(np_data)

print(labels)
