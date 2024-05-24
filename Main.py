import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Charger la base de données ORL
data = fetch_olivetti_faces()

# Pretraitement
images = data.images
targets = data.target
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Image {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
# Déterminer la nouvelle taille d'image
new_height = 32
new_width = 32

# Redimensionner les images
images_resized = np.array([np.resize(image, (new_height, new_width)) for image in images])

# Convertir les images en niveaux de gris si nécessaire
if images_resized.ndim == 4:
    gray_images = images_resized.mean(axis=3)
else:
    gray_images = images_resized

# Normaliser les valeurs des pixels entre 0 et 1
images_normalized = gray_images / 255.0

# Appliquer l'Analyse en Composantes Principales (PCA) directement sur les images
n_components = 100
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(images_normalized.reshape(len(images_normalized), -1))
print("Dimensions des données après PCA :", X_pca.shape)
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = (train_test_split(X_pca, targets, test_size=0.3, random_state=42))
# Définir la grille d'hyperparamètres
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
# Initialiser et entraîner GridSearchCV
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
# Afficher les meilleurs hyperparamètres trouvés par Grid Search
print("Meilleurs hyperparamètres:", grid_search.best_params_)

# Utiliser le meilleur modèle pour faire des prédictions sur l'ensemble de test
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
# Calculer la précision du meilleur modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle KNN avec Grid Search :", accuracy)

# Sélectionner un échantillon d'images de l'ensemble de test
sample_images = X_test[:10]
sample_labels_true = y_test[:10]
# Prédire les étiquettes de l'échantillon avec le meilleur modèle
sample_labels_pred = best_knn.predict(sample_images)
# Inverse de PCA pour obtenir les images originales
sample_images_restored = pca.inverse_transform(sample_images)
# Afficher les images avec les étiquettes prédites et réelles
plt.figure(figsize=(15, 6))
for i in range(len(sample_images_restored)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images_restored[i].reshape(new_height, new_width), cmap='gray')
    plt.title(f"True: {sample_labels_true[i]}\nPredicted: {sample_labels_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
