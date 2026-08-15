<!-- ---
layout: post
title: "Linear SVMs for NLP : TF-IDF, Hyperplans et Marges"
toc: true
author: Jeremie Mabiala
author_profile: "./assets/static/logo.jpeg"
summary: >-
  Nous expliquons pourquoi les SVM linéaires restent une méthode solide pour
  les tâches classiques de NLP : les textes deviennent des vecteurs TF-IDF,
  le classifieur apprend un hyperplan séparateur, et la marge est exactement
  une distance à cet hyperplan.
tags: [machine-learning, nlp, svm, linear-algebra, classification, python]
---

## Introduction

Dans beaucoup de tâches classiques de NLP, un texte est transformé en vecteur :
bag-of-words, TF-IDF, n-grammes, ou embeddings. Une fois cette transformation
faite, un document devient simplement un point $\mathbf{x} \in \mathbb{R}^d$,
où $d$ peut être très grand : un coefficient par mot, par n-gramme, ou par
dimension d'embedding.

Les **SVM linéaires** exploitent exactement cette représentation vectorielle.
Ils apprennent un hyperplan qui sépare deux classes, par exemple :

- avis positif vs avis négatif,
- spam vs non-spam,
- texte toxique vs texte acceptable,
- intention A vs intention B dans un chatbot,
- document médical vs document financier.

Le point clé est géométrique : la confiance du modèle est liée à la distance
du document à l'hyperplan séparateur.

<figure style="text-align:center; margin: 2rem 0;">
  <img src="{{ '/assets/images/linear_svm_nlp_margin.svg' | relative_url }}"
       alt="Pipeline NLP avec TF-IDF et SVM linéaire"
       style="max-width:820px; width:100%; border-radius:6px;
              box-shadow:0 2px 12px rgba(0,0,0,.12);">
  <figcaption style="margin-top:.6rem; font-size:.88rem; color:#666;">
    Un pipeline NLP classique : les documents sont convertis en vecteurs
    TF-IDF, puis un SVM linéaire apprend un hyperplan séparateur
    $\mathbf{w}^\top\mathbf{x} + b = 0$.
  </figcaption>
</figure>



## Représenter un texte comme un vecteur

Soit un vocabulaire de taille $d$ :

$$\mathcal{V} = \{t_1, t_2, \ldots, t_d\}.$$

Un document est représenté par un vecteur

$$\mathbf{x} = (x_1, x_2, \ldots, x_d)^\top \in \mathbb{R}^d,$$

où $x_j$ mesure l'importance du terme $t_j$ dans le document.

Avec une représentation **bag-of-words**, $x_j$ peut être le nombre
d'occurrences du mot $t_j$. Avec **TF-IDF**, $x_j$ combine deux idées :

- le mot doit être fréquent dans le document,
- mais il ne doit pas être trop fréquent dans tous les documents.

Une forme courante est :

$$\text{tf-idf}(t, d) = \text{tf}(t,d) \times \log\left(\frac{N}{\text{df}(t)}\right),$$

où $N$ est le nombre total de documents et $\text{df}(t)$ le nombre de
documents contenant le terme $t$.

Le résultat est souvent un vecteur **creux** : sur des dizaines de milliers de
mots possibles, un document n'en utilise qu'une petite fraction.



## Hyperplan de décision

Pour une classification binaire, on code les labels par

$$y_i \in \{-1, +1\}.$$

Un classifieur linéaire associe un score au document $\mathbf{x}$ :

$$s(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b.$$

La règle de décision est :

$$\hat{y} = \text{sign}\bigl(\mathbf{w}^\top \mathbf{x} + b\bigr).$$

L'ensemble des points pour lesquels le score vaut zéro forme l'hyperplan :

$$H = \{\mathbf{x} \in \mathbb{R}^d : \mathbf{w}^\top \mathbf{x} + b = 0\}.$$

Si $s(\mathbf{x}) > 0$, le document est placé du côté positif. Si
$s(\mathbf{x}) < 0$, il est placé du côté négatif.



## Distance à l'hyperplan

Dans le billet sur la
[distance d'un point à un hyperplan]({{ '/posts/distance-point-hyperplan' | relative_url }}),
on a vu que, pour

$$H = \{\mathbf{x} : \mathbf{a}^\top \mathbf{x} = c\},$$

la distance de $\mathbf{x}$ à $H$ est :

$$d(\mathbf{x}, H) = \frac{|\mathbf{a}^\top \mathbf{x} - c|}{\|\mathbf{a}\|}.$$

Pour un SVM linéaire, l'hyperplan s'écrit :

$$\mathbf{w}^\top \mathbf{x} + b = 0.$$

On obtient donc :

$$\boxed{d(\mathbf{x}, H) = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|}.}$$

La version signée est :

$$\delta(\mathbf{x}, H) = \frac{\mathbf{w}^\top \mathbf{x} + b}{\|\mathbf{w}\|}.$$

Son signe donne la classe prédite, et sa valeur absolue donne la distance au
frontière de décision.



## Marge du SVM

Le SVM ne cherche pas seulement un hyperplan qui sépare les classes. Il cherche
un hyperplan qui les sépare avec la plus grande **marge** possible.

Dans le cas séparable, on impose :

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1.$$

Les deux hyperplans de marge sont :

$$\mathbf{w}^\top \mathbf{x} + b = 1,$$

et

$$\mathbf{w}^\top \mathbf{x} + b = -1.$$

La distance entre ces deux hyperplans est :

$$\frac{2}{\|\mathbf{w}\|}.$$

Maximiser la marge revient donc à minimiser $\|\mathbf{w}\|$. C'est pourquoi
le SVM linéaire résout, dans sa forme souple :

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}}
\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i,$$

sous les contraintes

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i,\qquad \xi_i \geq 0.$$

Les variables $\xi_i$ autorisent quelques erreurs ou violations de marge. Le
paramètre $C$ contrôle le compromis entre une grande marge et peu d'erreurs
d'entraînement.



## Pourquoi c'est efficace en NLP

Les SVM linéaires sont particulièrement adaptés aux représentations TF-IDF.
La raison est simple : les vecteurs NLP classiques sont de très grande
dimension, mais ils sont creux (sparse en anglais).

Dans cet espace, certains mots ou n-grammes deviennent très discriminants :

- `excellent`, `incroyable`, `recommande` peuvent pousser vers une classe positive,
- `ennuyeux`, `terrible`, `remboursement` peuvent pousser vers une classe négative,
- `gratuit`, `gagnant`, `cliquez` peuvent pousser vers une classe spam.

Le vecteur $\mathbf{w}$ appris par le SVM donne un poids à chaque terme. Si
$w_j$ est positif, le terme $t_j$ pousse vers la classe $+1$. Si $w_j$ est
négatif, il pousse vers la classe $-1$.

Ainsi, le score

$$\mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^d w_j x_j + b$$

additionne les contributions des mots présents dans le document.



## Exemple Python

Voici un exemple minimal avec `scikit-learn`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

texts = [
    "film excellent avec un jeu d'acteurs formidable",
    "belle histoire et acteurs merveilleux",
    "mauvais film avec des scènes ennuyeuses",
    "scénario terrible et jeu d'acteurs médiocre",
]

labels = [1, 1, -1, -1]

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LinearSVC(C=1.0)
)

model.fit(texts, labels)

new_texts = [
    "film formidable et magnifique",
    "scénario ennuyeux avec un jeu d'acteurs médiocre",
]

predictions = model.predict(new_texts)
scores = model.decision_function(new_texts)

for text, y_hat, score in zip(new_texts, predictions, scores):
    print(f"{text!r}")
    print(f"prediction = {y_hat}, signed score = {score:.3f}")
```

`decision_function` retourne le score

$$\mathbf{w}^\top \mathbf{x} + b.$$

Pour obtenir la distance signée à l'hyperplan, on divise ce score par
$\|\mathbf{w}\|$ :

```python
import numpy as np

svm = model.named_steps["linearsvc"]
w = svm.coef_[0]
w_norm = np.linalg.norm(w)

signed_distances = scores / w_norm
print(signed_distances)
```

Plus la distance signée est grande en valeur absolue, plus le document est loin
de la frontière de décision. Une distance proche de zéro indique un exemple
ambigu, donc potentiellement difficile à classer.



## Interprétation des mots importants

Un avantage des modèles linéaires est leur lisibilité. On peut inspecter les
poids appris :

```python
vectorizer = model.named_steps["tfidfvectorizer"]
terms = vectorizer.get_feature_names_out()

top_positive = np.argsort(w)[-10:][::-1]
top_negative = np.argsort(w)[:10]

print("Positive terms")
for idx in top_positive:
    print(terms[idx], w[idx])

print("Negative terms")
for idx in top_negative:
    print(terms[idx], w[idx])
```

Les termes avec les grands poids positifs contribuent à la classe $+1$ ; ceux
avec les grands poids négatifs contribuent à la classe $-1$.

Cette interprétation est souvent plus directe qu'avec un modèle profond. Pour
des bases de textes modestes ou moyennes, un pipeline TF-IDF + SVM linéaire
reste donc une baseline très forte.



## Lien avec l'apprentissage actif

La distance à l'hyperplan donne aussi une stratégie simple pour sélectionner
les exemples à annoter.

Si

$$|\mathbf{w}^\top \mathbf{x} + b|$$

est grand, le modèle est loin de la frontière et semble confiant. Si cette
quantité est proche de zéro, le document est près de l'hyperplan : le modèle
hésite.

En apprentissage actif, on peut donc demander à un humain d'annoter les textes
les plus proches de l'hyperplan :

$$\mathbf{x}_{\text{next}} =
\arg\min_{\mathbf{x}} |\mathbf{w}^\top \mathbf{x} + b|.$$

Cette règle est simple, mais très naturelle : on annote en priorité les textes
qui peuvent le plus déplacer la frontière de décision.


En somme, dans un pipeline NLP classique, on a :

- un texte devient un vecteur $\mathbf{x}$ avec TF-IDF ou bag-of-words,
- un SVM linéaire apprend un hyperplan $\mathbf{w}^\top\mathbf{x} + b = 0$,
- le score $\mathbf{w}^\top\mathbf{x} + b$ donne le côté de l'hyperplan,
- la distance

$$\boxed{d(\mathbf{x}, H) = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|}}$$

mesure à quel point le document est éloigné de la frontière de décision.

C'est cette géométrie très simple qui rend les SVM linéaires si utiles en NLP :
ils transforment un problème linguistique en problème de séparation dans un
espace vectoriel de grande dimension. -->
