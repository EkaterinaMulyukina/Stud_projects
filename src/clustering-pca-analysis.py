import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

N = 250
np.random.seed(28)

cloud1 = np.random.normal(loc=[2, 3], scale=0.5, size=(N, 2))
cloud2 = np.random.normal(loc=[8, 1], scale=0.6, size=(N, 2))
cloud3 = np.random.normal(loc=[5, 7], scale=0.4, size=(N, 2))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(cloud1[:, 0], cloud1[:, 1], alpha=0.6, label='Облако 1')
plt.scatter(cloud2[:, 0], cloud2[:, 1], alpha=0.6, label='Облако 2')
plt.scatter(cloud3[:, 0], cloud3[:, 1], alpha=0.6, label='Облако 3')
plt.title('Начальные двумерные облака')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, alpha=0.3)

def extend_to_5d(cloud):

    x1 = cloud[:, 0]
    x2 = cloud[:, 1]

    epsilon = 1e-8
    x1_safe = x1 + epsilon

    x3 = x1 + x2
    x4 = np.log(x1_safe) + x2
    x5 = np.sin(x1 * x2)

    return np.column_stack([x1, x2, x3, x4, x5])

cloud1_5d = extend_to_5d(cloud1)
cloud2_5d = extend_to_5d(cloud2)
cloud3_5d = extend_to_5d(cloud3)

all_clouds_5d = np.vstack([cloud1_5d, cloud2_5d, cloud3_5d])

pca = PCA(n_components=2)
all_clouds_2d = pca.fit_transform(all_clouds_5d)

cloud1_2d = all_clouds_2d[:N]
cloud2_2d = all_clouds_2d[N:2*N]
cloud3_2d = all_clouds_2d[2*N:]

plt.subplot(1, 3, 2)
plt.scatter(cloud1_2d[:, 0], cloud1_2d[:, 1], alpha=0.6, label='Облако 1')
plt.scatter(cloud2_2d[:, 0], cloud2_2d[:, 1], alpha=0.6, label='Облако 2')
plt.scatter(cloud3_2d[:, 0], cloud3_2d[:, 1], alpha=0.6, label='Облако 3')
plt.title('Облака после PCA')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend()
plt.grid(True, alpha=0.3)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_clouds_2d)

silhouette_avg = silhouette_score(all_clouds_2d, cluster_labels)
inertia = kmeans.inertia_

silhouette_scores = []
inertias = []

for k in range(2, 6):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(all_clouds_2d)

    silhouette_scores.append(silhouette_score(all_clouds_2d, labels_temp))
    inertias.append(kmeans_temp.inertia_)

plt.subplot(1, 3, 3)
scatter = plt.scatter(all_clouds_2d[:, 0], all_clouds_2d[:, 1], c=cluster_labels, alpha=0.6)
plt.title(f'Кластеризация K-means')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(2, 6), silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров')
plt.ylabel('Коэффициент силуэта')
plt.title('Оптимальность количества кластеров')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(2, 6), inertias, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_k_silhouette = range(2, 6)[np.argmax(silhouette_scores)]

if best_k_silhouette == 3:
    print("Разделение на 3 кластера - оптимально")
else:
    print("Разделение на 3 кластера не оптимально")
