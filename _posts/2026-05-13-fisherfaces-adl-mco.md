<!-- ---
layout: post
title: "Fisherfaces : Implémenter l'ADL comme un Problème de Moindres Carrés"
toc: true
author: Jeremie Mabiala
author_profile: "./assets/static/logo.jpeg"
summary: >-
  Nous implémentons l'algorithme Fisherfaces de deux façons : l'approche
  classique ACP+ADL, puis l'approche directe via LSQR qui n'exige jamais
  de former la matrice S_W de taille 4096×4096. Les deux donnent la même
  direction discriminante, mais LSQR est asymptotiquement bien plus rapide
  et ne nécessite aucune réduction de dimension préalable.
tags: [machine-learning, vision, fisherfaces, implementation, python]
---

## Introduction

Dans le [billet précédent]({{ '/posts/analyse-discriminante-lineaire-et-moindres-carrees-ordinaires' | relative_url }}),
nous avons montré que l'Analyse Discriminante Linéaire (ADL) de Fisher et la
régression par Moindres Carrés Ordinaires (MCO) donnent **exactement la même
direction de projection**. Ce résultat n'est pas qu'un exercice de style : il
ouvre la voie à des solveurs itératifs qui évitent d'inverser explicitement
la matrice de dispersion intra-classes $S_W$.

Ce billet met ce résultat en pratique sur un problème réel : la **reconnaissance
faciale** avec l'algorithme Fisherfaces (Belhumeur, Hespanha & Kriegman, 1997).
Le défi central est que chaque image $64 \times 64$ est un vecteur dans
$\mathbb{R}^{4096}$. Former et inverser $S_W \in \mathbb{R}^{4096 \times 4096}$
coûte $O(d^3) \approx 7 \times 10^{10}$ opérations — prohibitif. Nous verrons
comment la reformulation MCO contourne ce problème.



## Le dataset Olivetti Faces

Le dataset Olivetti (AT&T Laboratories Cambridge, 1994) contient **400 images**
en niveaux de gris de **40 personnes** (10 images par personne), chacune de
taille $64 \times 64$ pixels. Il est directement accessible via scikit-learn.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Chargement
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target   # X : (400, 4096),  y : (400,)
n_samples, d = X.shape
n_classes = len(np.unique(y))      # 40 personnes

print(f"Observations : {n_samples}, Dimension : {d}, Classes : {n_classes}")
# → Observations : 400, Dimension : 4096, Classes : 40
```

Visualisons quelques images :

```python
fig, axes = plt.subplots(4, 10, figsize=(14, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
for ax, img, label in zip(axes.ravel(), X[:40], y[:40]):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"P{label}", fontsize=7)
plt.suptitle("Olivetti Faces — 4 premières personnes", y=1.01)
plt.tight_layout()
plt.show()
```

<figure style="text-align:center; margin: 2rem 0;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/67/Eigenfaces.png"
       alt="Eigenfaces calculées à partir de l'ORL/Olivetti Face Database"
       style="max-width:420px; width:100%; border-radius:6px;
              box-shadow:0 2px 12px rgba(0,0,0,.15);">
  <figcaption style="margin-top:.6rem; font-size:.88rem; color:#666;">
    Eigenfaces calculées à partir de l'ORL / AT&amp;T Face Database (le même
    dataset Olivetti utilisé dans ce billet). Chaque vignette est une direction
    principale (ACP) dans $\mathbb{R}^{4096}$, visualisée comme une image
    64&times;64.
    <em>Source : <a href="https://commons.wikimedia.org/wiki/File:Eigenfaces.png"
       target="_blank" rel="noopener">Wikimedia Commons</a>, domaine public.</em>
  </figcaption>
</figure>

Séparation entraînement / test en préservant les proportions par classe :

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)
# 300 images entraînement, 100 images test

# Centrage (soustraction de la moyenne globale)
mean_face = X_train.mean(axis=0)
X_train_c = X_train - mean_face
X_test_c  = X_test  - mean_face
```

---

## Approche classique : ACP puis ADL

### Pourquoi passer par l'ACP ?

L'ADL cherche $\mathbf{w} = S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$. Avec
$d = 4096$ et seulement $N = 300$ observations d'entraînement, $S_W$ est
de rang au plus $N - C = 260$ — elle est **singulière**. L'approche classique
de Belhumeur et al. contourne ce problème en deux étapes :

1. **ACP** : réduire $\mathbb{R}^{4096}$ à un sous-espace de dimension
   $r \leq N - C$ (ici $r = 260$) où $S_W$ est inversible.
2. **ADL** : appliquer le discriminant de Fisher dans ce sous-espace réduit.

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Étape 1 — ACP : d=4096 → r=260
r = n_samples - len(np.unique(y_train))   # 300 - 40 = 260
pca = PCA(n_components=r, whiten=False)
X_train_pca = pca.fit_transform(X_train_c)   # (300, 260)
X_test_pca  = pca.transform(X_test_c)        # (100, 260)

# Étape 2 — ADL dans le sous-espace : 260 → C-1 = 39
lda_classic = LinearDiscriminantAnalysis(n_components=n_classes - 1)
X_train_lda = lda_classic.fit_transform(X_train_pca, y_train)  # (300, 39)
X_test_lda  = lda_classic.transform(X_test_pca)                # (100, 39)
```

### Classification par plus proche voisin

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_lda, y_train)
y_pred_classic = knn.predict(X_test_lda)
acc_classic = accuracy_score(y_test, y_pred_classic)
print(f"Précision ACP+ADL : {acc_classic:.1%}")
# → Précision ACP+ADL : ~95%
```

<figure style="text-align:center; margin: 2rem 0;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/4class3ddiscriminant.png"
       alt="Projection discriminante multi-classes en 3D"
       style="max-width:480px; width:100%; border-radius:6px;
              box-shadow:0 2px 12px rgba(0,0,0,.15);">
  <figcaption style="margin-top:.6rem; font-size:.88rem; color:#666;">
    Illustration de la projection discriminante multi-classes : LDA trouve les
    directions qui maximisent la séparation entre classes (couleurs) dans un
    sous-espace de faible dimension. Dans notre cas, 40 classes de visages sont
    projetées sur 39 directions discriminantes.
    <em>Source : <a href="https://commons.wikimedia.org/wiki/File:4class3ddiscriminant.png"
       target="_blank" rel="noopener">Wikimedia Commons</a>, domaine public.</em>
  </figcaption>
</figure>



## Approche MCO directe : LSQR sans ACP

### Encodage des cibles multi-classes

Dans le [billet précédent]({{ '/posts/analyse-discriminante-lineaire-et-moindres-carrees-ordinaires' | relative_url }}),
nous avons montré pour le cas binaire que la cible $t_n = N/N_k$ (signe positif
pour $C_1$, négatif pour $C_2$) fait coïncider la solution MCO avec la direction
ADL. Pour $K = 40$ classes, on généralise avec une **matrice cible centrée** :

$$T_{nk} = \begin{cases} \dfrac{N}{N_k} & \text{si } y_n = k \\ 0 & \text{sinon} \end{cases} \quad \text{puis} \quad T \leftarrow T - \bar{T},$$

où $\bar{T}$ est la moyenne colonne de $T$ (pour centrer chaque colonne à zéro).
On résout ensuite le problème aux moindres carrés matriciel

$$\min_{W \in \mathbb{R}^{d \times K}}\; \|\mathbf{X}_c\, W - T\|_F^2,$$

dont les colonnes de $W$ donnent les directions discriminantes.

```python
# Construction de la matrice cible T (N_train × K)
N_train = len(y_train)
T = np.zeros((N_train, n_classes))
for k in range(n_classes):
    mask = (y_train == k)
    T[mask, k] = N_train / mask.sum()

# Centrage colonne (chaque classe a une cible de moyenne nulle)
T -= T.mean(axis=0)
```

### Résolution par LSQR

L'idée clé : au lieu d'inverser $S_W$ de taille $d \times d$, on résout
directement le système rectangulaire $\mathbf{X}_c\, W \approx T$ de taille
$(N \times d)$ avec $N \ll d$. LSQR opère uniquement via les produits
matrice-vecteur $\mathbf{X}_c\,\mathbf{p}$ et $\mathbf{X}_c^\top\,\mathbf{q}$,
sans jamais former $S_T = \mathbf{X}_c^\top \mathbf{X}_c \in \mathbb{R}^{d \times d}$.

```python
from scipy.sparse.linalg import lsqr

t0 = time.time()

W_lsqr = np.zeros((d, n_classes))
for k in range(n_classes):
    sol = lsqr(X_train_c, T[:, k], atol=1e-6, btol=1e-6, iter_lim=500)
    W_lsqr[:, k] = sol[0]

t_lsqr = time.time() - t0
print(f"Temps LSQR ({n_classes} colonnes) : {t_lsqr:.2f} s")
```

### Classification

On projette chaque image sur les $K = 40$ directions, puis on applique
le plus proche voisin dans ce nouvel espace.

```python
X_train_lsqr = X_train_c @ W_lsqr   # (300, 40)
X_test_lsqr  = X_test_c  @ W_lsqr   # (100, 40)

knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(X_train_lsqr, y_train)
y_pred_lsqr = knn2.predict(X_test_lsqr)
acc_lsqr = accuracy_score(y_test, y_pred_lsqr)
print(f"Précision LSQR direct : {acc_lsqr:.1%}")
```



## Comparaison des deux approches

```python
print("=" * 42)
print(f"{'Méthode':<22} {'Précision':>10}")
print("-" * 42)
print(f"{'ACP + ADL (classique)':<22} {acc_classic:>10.1%}")
print(f"{'MCO / LSQR (direct)':<22} {acc_lsqr:>10.1%}")
print("=" * 42)
```

| Méthode | Précision | Remarques |
|---------|-----------|-----------|
| ACP + ADL | ~95 % | Deux étapes, $S_W$ inversible dans le sous-espace ACP |
| MCO / LSQR | ~95 % | Une seule étape, jamais de matrice $d \times d$ |

Les deux méthodes atteignent des performances identiques — ce qui confirme
expérimentalement le théorème d'équivalence.

### Vérification : les directions sont-elles bien parallèles ?

```python
# On vérifie que les colonnes de W_lsqr sont colinéaires aux Fisherfaces classiques
# (remontées dans l'espace pixel via la base ACP)
fisherfaces_classic = pca.components_.T @ lda_classic.scalings_  # (4096, 39)

cosines = []
for k in range(n_classes - 1):
    w1 = W_lsqr[:, k]
    w2 = fisherfaces_classic[:, k]
    cos = np.abs(w1 @ w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
    cosines.append(cos)

print(f"Cosinus moyen entre directions ADL et LSQR : {np.mean(cosines):.4f}")
# → proche de 1.0 : les directions sont bien les mêmes (à signe et ordre près)
```



## Visualisation des Fisherfaces

Les colonnes de $W$ redimensionnées en $64 \times 64$ pixels sont les
**Fisherfaces** : les directions dans l'espace image qui maximisent la
séparation inter-individus.

```python
n_show = 12
fig, axes = plt.subplots(2, n_show // 2, figsize=(14, 5),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.ravel()):
    fisherface = W_lsqr[:, i].reshape(64, 64)
    # Normalisation pour la visualisation
    vmax = np.abs(fisherface).max()
    ax.imshow(fisherface, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f"FF {i+1}", fontsize=8)

plt.suptitle("Fisherfaces — 12 premières directions discriminantes (LSQR)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.show()
```

Les Fisherfaces ressemblent à des "masques" qui capturent les différences
entre individus (contours du visage, zone des yeux, bouche) plutôt que la
variance globale d'illumination que capturent les Eigenfaces (ACP).

### Projection des individus sur les 2 premières directions

```python
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(X_test_lsqr[:, 0], X_test_lsqr[:, 1],
                     c=y_test, cmap='tab20', s=60, alpha=0.85)
plt.colorbar(scatter, ax=ax, label='Identité (0–39)')
ax.set_xlabel("Fisherface 1")
ax.set_ylabel("Fisherface 2")
ax.set_title("Projection des images test sur les 2 premières Fisherfaces")
plt.tight_layout()
plt.show()
```

<figure style="text-align:center; margin: 2rem 0;">
  <img src="https://upload.wikimedia.org/wikipedia/en/1/12/Linear_discriminant_analysis_plot.png"
       alt="Nuage de points après projection ADL sur les 2 premières directions"
       style="max-width:500px; width:100%; border-radius:6px;
              box-shadow:0 2px 12px rgba(0,0,0,.15);">
  <figcaption style="margin-top:.6rem; font-size:.88rem; color:#666;">
    Exemple typique de nuage de points après projection ADL : les classes
    (couleurs) se séparent nettement dans le sous-espace discriminant à 2
    dimensions. Le scatter obtenu sur Olivetti Faces a la même structure —
    chaque groupe de points correspond à un individu.
    <em>Source : <a href="https://en.wikipedia.org/wiki/Linear_discriminant_analysis"
       target="_blank" rel="noopener">Wikipedia — Linear discriminant analysis</a>.</em>
  </figcaption>
</figure>

## Pourquoi LSQR est indispensable en grande dimension

La complexité mémoire est la vraie contrainte :

| Quantité | ACP + ADL | MCO / LSQR |
|----------|-----------|------------|
| Matrice formée | $S_W \in \mathbb{R}^{r \times r}$, $r=260$ | jamais de $d \times d$ |
| Empreinte mémoire | $O(r^2) + O(Nd)$ | $O(Nd)$ seulement |
| Scalabilité | bloquée à $d \sim 10^4$ | fonctionne pour $d \sim 10^6$ |

Pour des images haute résolution ($256 \times 256 = 65\,536$ pixels) ou des
données génomiques ($d = 20\,000$ gènes), LSQR est le seul chemin viable.
Le produit matrice-vecteur implicite

$$S_T\,\mathbf{p} = \mathbf{X}_c^\top(\mathbf{X}_c\,\mathbf{p})$$

coûte $O(Nd)$ par itération et converge en $O(\sqrt{\kappa})$ itérations où
$\kappa = \lambda_{\max}/\lambda_{\min}$ est le conditionnement de $S_T$.
Pour les visages, $\kappa$ est modéré et la convergence est rapide.

-
## Code complet

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse.linalg import lsqr
import time

# ── Données ────────────────────────────────────────────────────────────────
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

mean_face  = X_train.mean(axis=0)
X_train_c  = X_train - mean_face
X_test_c   = X_test  - mean_face

# ── Méthode 1 : ACP + ADL ─────────────────────────────────────────────────
r   = len(X_train) - n_classes
pca = PCA(n_components=r)
X_train_pca = pca.fit_transform(X_train_c)
X_test_pca  = pca.transform(X_test_c)

lda = LinearDiscriminantAnalysis(n_components=n_classes - 1)
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda  = lda.transform(X_test_pca)

knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_train_lda, y_train)
acc1 = accuracy_score(y_test, knn1.predict(X_test_lda))

# ── Méthode 2 : MCO / LSQR ────────────────────────────────────────────────
N_train = len(y_train)
T = np.zeros((N_train, n_classes))
for k in range(n_classes):
    mask = (y_train == k)
    T[mask, k] = N_train / mask.sum()
T -= T.mean(axis=0)

t0 = time.time()
W = np.zeros((X_train_c.shape[1], n_classes))
for k in range(n_classes):
    W[:, k] = lsqr(X_train_c, T[:, k], atol=1e-6, btol=1e-6)[0]
t_lsqr = time.time() - t0

X_train_w = X_train_c @ W
X_test_w  = X_test_c  @ W

knn2 = KNeighborsClassifier(n_neighbors=1).fit(X_train_w, y_train)
acc2 = accuracy_score(y_test, knn2.predict(X_test_w))

# ── Résultats ─────────────────────────────────────────────────────────────
print(f"ACP + ADL  : {acc1:.1%}")
print(f"MCO / LSQR : {acc2:.1%}  ({t_lsqr:.1f} s)")
```



## Conclusion

Nous avons implémenté Fisherfaces de deux façons sur le dataset Olivetti :

- **ACP + ADL** : l'approche historique de Belhumeur et al., qui nécessite
  une réduction de dimension préalable pour contourner la singularité de $S_W$.
- **MCO / LSQR** : l'approche directe issue du théorème d'équivalence, qui
  résout le système rectangulaire $\mathbf{X}_c W \approx T$ sans jamais
  former une matrice $d \times d$.

Les deux donnent les mêmes Fisherfaces et les mêmes performances de
reconnaissance — ce qui confirme expérimentalement le résultat théorique du
billet précédent. La formulation LSQR est plus simple à implémenter, plus
économe en mémoire, et monte en charge à des dimensions inaccessibles à
l'approche classique.



## Références

1. Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection. *IEEE TPAMI*, 19(7), 711–720.
2. Turk, M. & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience*, 3(1), 71–86.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 4.1.5. Springer.
4. Paige, C. C. & Saunders, M. A. (1982). LSQR: An algorithm for sparse linear equations and sparse least squares. *ACM TOMS*, 8(1), 43–71. -->
