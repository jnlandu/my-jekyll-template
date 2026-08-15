<!-- ---
layout: post
title: "L'Analyse Discriminante Linéaire comme un Problème de Moindres Carrés"
toc: true
author: Jeremie Mabiala
author_profile: "./assets/static/logo.jpeg"
summary: >-
  Nous montrons que l'Analyse Discriminante Linéaire de Fisher et la régression
  par Moindres Carrés Ordinaires partagent la même direction de projection optimale
  lorsque les étiquettes de classe sont encodées de manière appropriée. Cette
  équivalence ouvre la voie aux solveurs itératifs — tels que les Gradients
  Conjugués et LSQR — comme alternatives à la formule directe classique
  w = S_W^{-1}(m_1 - m_2).
tags: [machine-learning, algebre-lineaire, classification, optimisation]
--- -->

<!-- ## Introduction

L'Analyse Discriminante Linéaire (ADL) et la régression par Moindres Carrés
Ordinaires (MCO) semblent être des outils complètement différents. L'ADL est
un classifieur : elle trouve une projection qui sépare au mieux deux classes
(ou plus). Les MCO sont un outil de régression : ils ajustent une fonction
linéaire à des cibles continues en minimisant les résidus au carré.

Pourtant, sous la surface, ils résolvent le même système linéaire. Ce billet
donne une dérivation autonome du fait que, pour la classification binaire,
la solution MCO avec des étiquettes de classe encodées de façon appropriée
donne **exactement la même direction de projection** que l'ADL de Fisher.
En chemin, nous verrons comment les matrices de dispersion apparaissent
naturellement à partir des équations normales de la régression, et pourquoi
cette équivalence a une importance pratique.



## L'Analyse Discriminante Linéaire de Fisher

### Mise en place

Soit $$\{\mathbf{x}_n, c_n\}_{n=1}^{N}$$ un jeu de données étiqueté dans
$$\mathbb{R}^d$$ avec deux classes $$C_1$$ et $$C_2$$, contenant respectivement
$N_1$ et $N_2$ observations ($N_1 + N_2 = N$). On définit les moyennes de
classe

$$\mathbf{m}_k = \frac{1}{N_k}\sum_{n \in C_k} \mathbf{x}_n, \qquad k = 1, 2,$$

et la moyenne globale $\mathbf{m} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{x}_n = \frac{N_1}{N}\mathbf{m}_1 + \frac{N_2}{N}\mathbf{m}_2$.


### Le critère de Fisher

On cherche une direction $\mathbf{w} \in \mathbb{R}^d$ telle que les
projections scalaires $y_n = \mathbf{w}^{\top}\mathbf{x}_n$ soient bien
séparées entre les classes. Fisher formalise ceci en maximisant le rapport

$$J(\mathbf{w}) = \frac{\mathbf{w}^{\top} S_B\, \mathbf{w}}{\mathbf{w}^{\top} S_W\, \mathbf{w}},$$

où la **matrice de dispersion inter-classes** est

$$S_B = (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^{\top},$$

et la **matrice de dispersion intra-classes** est

$$S_W = \sum_{k=1}^{2}\sum_{n \in C_k}(\mathbf{x}_n - \mathbf{m}_k)(\mathbf{x}_n - \mathbf{m}_k)^{\top}.$$

### La solution de l'ADL

En utilisant les multiplicateurs de Lagrange (ou en différenciant $J$ et en
annulant le gradient), la direction optimale satisfait le problème aux valeurs
propres généralisées

$$S_B\,\mathbf{w} = \lambda\, S_W\,\mathbf{w}.$$

Puisque $S_B\,\mathbf{w}$ pointe toujours dans la direction de
$(\mathbf{m}_1 - \mathbf{m}_2)$, la solution se réduit à

$$\boxed{\mathbf{w}^* \propto S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2).}$$

Le classifieur projette chaque point sur $\mathbf{w}^*$ et l'affecte à la
classe dont la moyenne projetée est la plus proche.

<figure style="text-align:center; margin: 2rem 0;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/f4/Fisher2classes.png"
       alt="Discriminant linéaire de Fisher — deux classes"
       style="max-width:580px; width:100%; border-radius:6px; box-shadow:0 2px 12px rgba(0,0,0,.18);">
  <figcaption style="margin-top:.6rem; font-size:.9rem; color:#888;">
    <strong>Figure 1.</strong> Projection de Fisher sur deux classes : la droite
    rouge maximise la séparation inter-classes tout en minimisant la dispersion
    intra-classes. Source : Wikimedia Commons (CC BY-SA 4.0).
  </figcaption>
</figure>

## Les Moindres Carrés : une perspective par régression

### Encodage des étiquettes de classe

L'idée clé est de remplacer les étiquettes discrètes $c_n \in \{C_1, C_2\}$
par des cibles à valeurs réelles

$$t_n = \begin{cases} \dfrac{N}{N_1} & \text{si } \mathbf{x}_n \in C_1, \\[6pt] -\dfrac{N}{N_2} & \text{si } \mathbf{x}_n \in C_2. \end{cases}$$

Cet encodage n'est pas arbitraire. Une propriété fondamentale est que les
cibles sont **de moyenne nulle** :

$$\sum_{n=1}^{N} t_n = N_1 \cdot \frac{N}{N_1} + N_2 \cdot \left(-\frac{N}{N_2}\right) = N - N = 0.$$

### L'objectif MCO

On ajuste un modèle affine $\hat{y}_n = \mathbf{w}^{\top}\mathbf{x}_n + w_0$
en minimisant la somme des résidus au carré :

$$E(\mathbf{w}, w_0) = \frac{1}{2}\sum_{n=1}^{N}\left(\mathbf{w}^{\top}\mathbf{x}_n + w_0 - t_n\right)^2.$$



## Dérivation des équations normales

### Résolution du biais $w_0$

En posant $\partial E / \partial w_0 = 0$ :

$$\sum_{n=1}^{N}\left(\mathbf{w}^{\top}\mathbf{x}_n + w_0 - t_n\right) = 0.$$

En réarrangeant et en utilisant $\sum_n t_n = 0$ et $\sum_n \mathbf{x}_n = N\mathbf{m}$ :

$$N w_0 + \mathbf{w}^{\top}(N\mathbf{m}) = 0 \implies \boxed{w_0 = -\mathbf{w}^{\top}\mathbf{m}.}$$

### Résolution du vecteur de poids $\mathbf{w}$

En posant $\partial E / \partial \mathbf{w} = 0$ :

$$\sum_{n=1}^{N}\left(\mathbf{w}^{\top}\mathbf{x}_n + w_0 - t_n\right)\mathbf{x}_n = \mathbf{0}.$$

En substituant $w_0 = -\mathbf{w}^{\top}\mathbf{m}$ :

$$\sum_{n=1}^{N}\left(\mathbf{w}^{\top}(\mathbf{x}_n - \mathbf{m}) - t_n\right)\mathbf{x}_n = \mathbf{0}.$$

En développant :

$$\left(\sum_{n=1}^{N}\mathbf{x}_n\mathbf{x}_n^{\top}\right)\mathbf{w}
  - N\mathbf{m}\mathbf{m}^{\top}\mathbf{w}
  = \sum_{n=1}^{N} t_n \mathbf{x}_n.$$

La matrice à gauche est la **matrice de dispersion totale** :

$$S_T = \sum_{n=1}^{N}(\mathbf{x}_n - \mathbf{m})(\mathbf{x}_n - \mathbf{m})^{\top}
     = \sum_{n=1}^{N}\mathbf{x}_n\mathbf{x}_n^{\top} - N\mathbf{m}\mathbf{m}^{\top}.$$

À droite, en calculant $\sum_n t_n \mathbf{x}_n$ :

$$\sum_{n=1}^{N} t_n \mathbf{x}_n
  = \frac{N}{N_1}\sum_{n \in C_1}\mathbf{x}_n - \frac{N}{N_2}\sum_{n \in C_2}\mathbf{x}_n
  = \frac{N}{N_1}\cdot N_1\mathbf{m}_1 - \frac{N}{N_2}\cdot N_2\mathbf{m}_2
  = N(\mathbf{m}_1 - \mathbf{m}_2).$$

L'équation normale devient donc :

$$\boxed{S_T\,\mathbf{w} = N(\mathbf{m}_1 - \mathbf{m}_2).}$$

<figure style="text-align:center; margin: 2rem 0;">
  <img src="https://upload.wikimedia.org/wikipedia/en/1/12/Linear_discriminant_analysis_plot.png"
       alt="Frontière de décision ADL et ellipses de dispersion"
       style="max-width:540px; width:100%; border-radius:6px; box-shadow:0 2px 12px rgba(0,0,0,.18);">
  <figcaption style="margin-top:.6rem; font-size:.9rem; color:#888;">
    <strong>Figure 2.</strong> Frontière de décision ADL (droite noire) et
    ellipses de dispersion des deux classes. La normale à cette frontière est
    précisément la direction $\mathbf{w}^*$ que nos équations normales retrouvent.
    Source : Wikimedia Commons (CC BY-SA 4.0).
  </figcaption>
</figure>

---

## Lien entre $S_T$, $S_W$ et $S_B$

La matrice de dispersion totale se décompose comme suit :

$$S_T = S_W + S_B^{(\text{total})},$$

où la **dispersion inter-classes totale** est

$$S_B^{(\text{total})} = \sum_{k=1}^{2} N_k(\mathbf{m}_k - \mathbf{m})(\mathbf{m}_k - \mathbf{m})^{\top}.$$

Montrons que $S_B^{(\text{total})}\,\mathbf{w}$ est proportionnel à
$(\mathbf{m}_1 - \mathbf{m}_2)$ et ne modifie donc pas la *direction* de $\mathbf{w}$.

En notant que $\mathbf{m} = \frac{N_1}{N}\mathbf{m}_1 + \frac{N_2}{N}\mathbf{m}_2$, on a :

$$\mathbf{m}_1 - \mathbf{m} = \frac{N_2}{N}(\mathbf{m}_1 - \mathbf{m}_2),
  \qquad \mathbf{m}_2 - \mathbf{m} = -\frac{N_1}{N}(\mathbf{m}_1 - \mathbf{m}_2).$$

Par conséquent :

$$S_B^{(\text{total})}\,\mathbf{w}
  = N_1\frac{N_2^2}{N^2}\left[(\mathbf{m}_1-\mathbf{m}_2)(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}\right]
  + N_2\frac{N_1^2}{N^2}\left[(\mathbf{m}_1-\mathbf{m}_2)(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}\right].$$

En regroupant les termes :

$$S_B^{(\text{total})}\,\mathbf{w}
  = \frac{N_1 N_2}{N^2}(N_1 + N_2)\left[(\mathbf{m}_1-\mathbf{m}_2)(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}\right]
  = \frac{N_1 N_2}{N}\,S_B\,\mathbf{w}.$$

Donc $S_B^{(\text{total})}\,\mathbf{w}$ est bien dans la direction de
$(\mathbf{m}_1 - \mathbf{m}_2)$. Notons $\alpha$ le scalaire
$\frac{N_1 N_2}{N}(\mathbf{m}_1-\mathbf{m}_2)^{\top}\mathbf{w}$. Alors :

$$S_T\,\mathbf{w} = S_W\,\mathbf{w} + \alpha\,(\mathbf{m}_1 - \mathbf{m}_2).$$

En substituant dans l'équation normale :

$$S_W\,\mathbf{w} + \alpha\,(\mathbf{m}_1 - \mathbf{m}_2) = N(\mathbf{m}_1 - \mathbf{m}_2).$$

En réarrangeant :

$$S_W\,\mathbf{w} = (N - \alpha)(\mathbf{m}_1 - \mathbf{m}_2).$$

Le scalaire $(N - \alpha)$ est une constante positive qui n'affecte pas la
direction de $\mathbf{w}$, donc :

$$\mathbf{w} \propto S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2).$$

C'est précisément la solution de l'ADL de Fisher.



## Théorème d'équivalence

> **Théorème.** Pour une classification binaire avec $N_1$ observations dans
> $C_1$ et $N_2$ dans $C_2$, avec les cibles $t_n = N/N_1$ pour $C_1$ et
> $t_n = -N/N_2$ pour $C_2$, le vecteur de poids $$\mathbf{w}^*$$ qui minimise
> $E(\mathbf{w}, w_0) = \frac{1}{2}\Vert \mathbf{X}\mathbf{w} + w_0\mathbf{1} - \mathbf{t}\Vert^2$
> vérifie $\mathbf{w}^* \propto S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$,
> qui est aussi la direction de l'ADL de Fisher.

La preuve est la dérivation des sections 3 et 4. Remarquons que les deux
méthodes s'accordent sur la **direction de projection** mais pas sur
l'intercept ni sur l'échelle de $\mathbf{w}$ : les MCO donnent un modèle
affine spécifique, tandis que l'ADL donne une direction pour un classifieur
au plus proche voisin des moyennes.

<figure style="text-align:center; margin:2rem 0;">
<svg width="500" height="300" viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg"
     style="font-family:Georgia,'Times New Roman',serif; max-width:100%;">

  <rect width="500" height="300" fill="#f8f9fa" rx="10" stroke="#ddd" stroke-width="1"/>

  <line x1="50" y1="150" x2="450" y2="150" stroke="#e4e4e4" stroke-width="1"/>
  <line x1="245" y1="20"  x2="245" y2="280" stroke="#e4e4e4" stroke-width="1"/>

  <ellipse cx="140" cy="205" rx="68" ry="32"
           transform="rotate(-22 140 205)"
           fill="rgba(192,57,43,0.13)" stroke="#c0392b" stroke-width="1.5"/>
  <text x="90" y="210" font-size="14" fill="#c0392b" font-style="italic">C₂</text>

  <ellipse cx="350" cy="97" rx="68" ry="32"
           transform="rotate(-22 350 97)"
           fill="rgba(41,128,185,0.13)" stroke="#2980b9" stroke-width="1.5"/>
  <text x="390" y="95" font-size="14" fill="#2980b9" font-style="italic">C₁</text>

  <circle cx="140" cy="205" r="5" fill="#c0392b"/>
  <text x="118" y="226" font-size="12" fill="#c0392b">m₂</text>

  <circle cx="350" cy="97"  r="5" fill="#2980b9"/>
  <text x="356" y="92"  font-size="12" fill="#2980b9">m₁</text>

  <circle cx="245" cy="151" r="4" fill="#555"/>


  <defs>
    <marker id="mkg" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#7f8c8d"/>
    </marker>
    <marker id="mkt" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0,8 3,0 6" fill="#16a085"/>
    </marker>
  </defs>

  <line x1="245" y1="151" x2="338" y2="104"
        stroke="#7f8c8d" stroke-width="2" stroke-dasharray="7,4"
        marker-end="url(#mkg)"/>
  <text font-size="11" fill="#7f8c8d" transform="rotate(-27 285 120)">
    <tspan x="272" y="128">m₁ − m₂</tspan>
  </text>


  <line x1="121" y1="191" x2="245" y2="151"
        stroke="#16a085" stroke-width="2.5"/>
  <line x1="245" y1="151" x2="369" y2="111"
        stroke="#16a085" stroke-width="2.5" marker-end="url(#mkt)"/>
  <text x="374" y="107" font-size="13" fill="#16a085" font-weight="bold">w*</text>

  <path d="M 275,141 A 32,32 0 0,0 285,130"
        stroke="#555" stroke-width="1.3" fill="none"/>
  <text x="284" y="140" font-size="10" fill="#555">θ</text>


  <line x1="216" y1="61" x2="274" y2="241"
        stroke="#bbb" stroke-width="1.5" stroke-dasharray="5,5"/>
  <text x="275" y="247" font-size="10" fill="#aaa">frontière</text>

  <rect x="10" y="12" width="220" height="68" rx="7"
        fill="white" stroke="#16a085" stroke-width="1.5"/>
  <text x="115" y="31" text-anchor="middle" font-size="11" fill="#16a085" font-weight="bold">
    Théorème d'équivalence
  </text>
  <text x="115" y="49" text-anchor="middle" font-size="11" fill="#333">
    w*_ADL  =  w*_MCO
  </text>
  <text x="115" y="67" text-anchor="middle" font-size="10.5" fill="#555">
    ∝  S_W⁻¹ (m₁ − m₂)
  </text>

  <line x1="10" y1="258" x2="45" y2="258" stroke="#7f8c8d" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="50" y="262" font-size="10" fill="#7f8c8d">direction m₁ − m₂ (naive)</text>
  <line x1="10" y1="278" x2="45" y2="278" stroke="#16a085" stroke-width="2.5"/>
  <text x="50" y="282" font-size="10" fill="#16a085">direction optimale w* (ADL = MCO)</text>

</svg>
<figcaption style="margin-top:.7rem; font-size:.88rem; color:#888;">
  <strong>Figure 4.</strong> Géométrie du théorème d'équivalence. La direction naïve
  m₁ − m₂ (tirets gris) et la direction optimale w* = S_W⁻¹(m₁ − m₂) (trait teal)
  sont en général distinctes (angle θ ≠ 0) lorsque S_W est non sphérique.
  Le théorème affirme que l'ADL de Fisher et les MCO aboutissent tous deux à <strong>w*</strong>.
</figcaption>
</figure> -->



<!-- ## Forme matricielle et solveurs itératifs

### Équation normale compacte

En notant $\mathbf{X} \in \mathbb{R}^{N \times d}$ la matrice de données
(les lignes sont les observations) et $\mathbf{t} \in \mathbb{R}^N$ le vecteur
cible, le problème MCO s'écrit :

$$\min_{\mathbf{w}, w_0}\; \frac{1}{2}\|\tilde{\mathbf{X}}\boldsymbol{\beta} - \mathbf{t}\|^2,
\qquad \tilde{\mathbf{X}} = [\mathbf{X} \;\; \mathbf{1}],
\quad \boldsymbol{\beta} = \begin{pmatrix}\mathbf{w} \\ w_0\end{pmatrix}.$$

L'équation normale est :

$$\tilde{\mathbf{X}}^{\top}\tilde{\mathbf{X}}\,\boldsymbol{\beta} = \tilde{\mathbf{X}}^{\top}\mathbf{t}.$$ -->

<!-- ### Pourquoi cela importe : l'ouverture vers les méthodes itératives

La formule ADL $\mathbf{w} = S_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$ est
habituellement résolue par factorisation de $S_W$ (par exemple, décomposition
de Cholesky), ce qui coûte $O(d^3)$. En grande dimension, c'est coûteux et
parfois numériquement instable.

La formulation MCO reformule le même problème comme un système aux moindres
carrés standard, que l'on peut résoudre avec :

| Méthode | Coût par itération | Adapté pour |
|---------|-------------------|-------------|
| Cholesky (direct) | $O(d^3)$ une fois | petit $d$ |
| **Gradients Conjugués (GC)** | $O(Nd)$ | grande $S_W$ creuse |
| **LSQR** | $O(Nd)$ | systèmes rectangulaires, stabilité numérique |
| **SVD aléatoire** | $O(Nd\,k)$ | cas de faible rang approché |

Pour les problèmes à grande échelle où $d$ (dimension des données) est de
l'ordre des milliers, un solveur itératif comme **LSQR** ou les **GC appliqués
à $S_T$** peut converger en bien moins de $d$ itérations lorsque la matrice de
dispersion a des valeurs propres à décroissance rapide — situation courante en pratique.

### L'approche GC explicitement

En appliquant les GC au système $S_T\,\mathbf{w} = N(\mathbf{m}_1 - \mathbf{m}_2)$ :

1. Initialiser $\mathbf{w}_0 = \mathbf{0}$, résidu $\mathbf{r}_0 = N(\mathbf{m}_1 - \mathbf{m}_2)$, direction $\mathbf{p}_0 = \mathbf{r}_0$.
2. Pour $k = 0, 1, 2, \ldots$ jusqu'à convergence :

$$\alpha_k = \frac{\mathbf{r}_k^{\top}\mathbf{r}_k}{\mathbf{p}_k^{\top} S_T\,\mathbf{p}_k},
\qquad \mathbf{w}_{k+1} = \mathbf{w}_k + \alpha_k\,\mathbf{p}_k,$$

$$\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k\, S_T\,\mathbf{p}_k,
\qquad \beta_k = \frac{\mathbf{r}_{k+1}^{\top}\mathbf{r}_{k+1}}{\mathbf{r}_k^{\top}\mathbf{r}_k},
\qquad \mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k\,\mathbf{p}_k.$$

Chaque produit matrice-vecteur $S_T\,\mathbf{p}$ peut être calculé
implicitement à partir des données comme $S_T\,\mathbf{p} = \mathbf{X}_c^{\top}(\mathbf{X}_c\,\mathbf{p})$,
où $\mathbf{X}_c$ est la matrice de données centrée. Cela maintient
l'empreinte mémoire à $O(Nd)$ plutôt que $O(d^2)$.



## Exemple numérique

Supposons $d = 2$, $N_1 = N_2 = 50$, et

$$S_W = \begin{pmatrix}2 & 1\\ 1 & 3\end{pmatrix},
\qquad \mathbf{m}_1 - \mathbf{m}_2 = \begin{pmatrix}1 \\ 0\end{pmatrix}.$$

**ADL (directe) :**

$$\mathbf{w}^* = S_W^{-1}\begin{pmatrix}1\\0\end{pmatrix}
  = \frac{1}{5}\begin{pmatrix}3 & -1\\ -1 & 2\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix}
  = \frac{1}{5}\begin{pmatrix}3\\ -1\end{pmatrix}.$$

**MCO (via équation normale) :** Pour des classes équilibrées ($N_1 = N_2$),
l'encodage des cibles se simplifie en $t_n = +1$ pour $C_1$ et $t_n = -1$
pour $C_2$. En résolvant $S_T\,\mathbf{w} = N(\mathbf{m}_1 - \mathbf{m}_2)$
et en utilisant $S_T = S_W + S_B$ (toutes deux calculables à partir des
données), on retrouve la même direction $\propto (3, -1)^{\top}$. -->

<!-- <figure style="text-align:center; margin: 2rem 0;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/4class3ddiscriminant.png"
       alt="ADL à 4 classes en 3D"
       style="max-width:520px; width:100%; border-radius:6px; box-shadow:0 2px 12px rgba(0,0,0,.18);">
  <figcaption style="margin-top:.6rem; font-size:.9rem; color:#888;">
    <strong>Figure 3.</strong> Extension à 4 classes en 3D : chaque axe
    discriminant est obtenu par la même formule $\mathbf{w}_k \propto S_W^{-1}(\mathbf{m}_k - \mathbf{m})$,
    ou de façon équivalente, par MCO avec un encodage one-vs-all.
    Source : Wikimedia Commons (CC BY-SA 4.0).
  </figcaption>
</figure> -->

<!-- ## Applications concrètes

L'équivalence ADL–MCO change la façon dont on implante et étend l'ADL dans
des domaines très différents.

### Reconnaissance faciale — Fisherfaces

L'algorithme **Fisherfaces** (Belhumeur, Hespanha & Kriegman, 1997) applique
l'ADL directement aux pixels d'images de visages. Une image $64 \times 64$
est un vecteur dans $\mathbb{R}^{4096}$. L'inversion directe de $S_W \in
\mathbb{R}^{4096 \times 4096}$ coûte $O(d^3) \approx 7 \times 10^{10}$
opérations — inaccessible.

La solution classique passe d'abord par une ACP pour réduire la dimension,
puis applique l'ADL dans le sous-espace réduit. Avec la reformulation MCO,
on peut à la place appliquer **LSQR** directement sur les pixels centrés : le
produit $S_T\,\mathbf{p} = \mathbf{X}_c^\top(\mathbf{X}_c\,\mathbf{p})$ ne
matérialise jamais la matrice $4096 \times 4096$ et converge en quelques
dizaines d'itérations.

### Génomique — classification de sous-types de cancer

En RNA-seq ou microarray, on dispose typiquement de $N \approx 200$
échantillons et $d \approx 20\,000$ gènes, donc $d \gg N$ et $S_W$ est
singulière. La formulation MCO suggère immédiatement la correction : ajouter
un terme de régularisation $\lambda I$ à $S_T$, ce qui revient à résoudre

$$\min_{\mathbf{w}}\; \tfrac{1}{2}\|\mathbf{X}_c\,\mathbf{w} - \mathbf{t}\|^2 + \tfrac{\lambda}{2}\|\mathbf{w}\|^2,$$

dont la solution $(S_T + \lambda I)^{-1}N(\mathbf{m}_1-\mathbf{m}_2)$ reste
stable même quand $S_T$ est singulière. C'est exactement l'ADL régularisée
(RDA) de Friedman (1989), obtenue ici par simple inspection du problème de
régression.

### Finance — le Z-score d'Altman

En 1968, Edward Altman publie un modèle de prédiction de faillite d'entreprise
basé sur cinq ratios financiers $(x_1, \ldots, x_5)$ :

$$Z = 1.2\,x_1 + 1.4\,x_2 + 3.3\,x_3 + 0.6\,x_4 + 1.0\,x_5.$$

C'est un discriminant linéaire pur : les coefficients sont obtenus par ADL
sur $N = 66$ entreprises dans $\mathbb{R}^5$. Une entreprise avec $Z < 1.81$
est classée à risque, $Z > 2.99$ est saine. Plus de 50 ans après, ce modèle
reste une référence dans l'industrie financière — illustration de la
robustesse de la solution MCO en faible dimension.

### Spectroscopie — chimiométrie

En spectroscopie proche infrarouge (NIR), un spectre est un vecteur de
$d \approx 1\,000$–$10\,000$ longueurs d'onde fortement corrélées. L'objectif
est de classer des substances (type d'huile, qualité du grain, contrefaçon
alimentaire). La quasi-singularité de $S_W$ rend l'inversion directe instable ;
la formulation MCO permet d'utiliser **LSQR** ou la régression ridge, qui
exploitent la structure de faible rang effectif pour converger rapidement. En
pratique, les solveurs itératifs issus de l'interprétation MCO offrent des
gains d'un ordre de grandeur par rapport à l'inversion directe.

---

## Conclusion

L'équivalence établie ici peut s'énoncer simplement :

> **L'ADL et les MCO partagent la même direction de projection optimale.**

Plus précisément, si l'on encode les appartenances aux classes par
$t_n = N/N_k$ (positif pour une classe, négatif pour l'autre), le vecteur
de poids de la régression MCO satisfait exactement le même système linéaire
que le discriminant de Fisher. La dérivation passe par la décomposition
$S_T = S_W + S_B$ et l'observation que $S_B\,\mathbf{w}$ est toujours
colinéaire à la différence des moyennes de classe $(\mathbf{m}_1 - \mathbf{m}_2)$.

C'est plus qu'une curiosité mathématique :

- Cela donne un **nouvel algorithme** pour l'ADL : résoudre l'équation
  normale des MCO avec n'importe quelle méthode — directe ou itérative.
- Cela permet une **ADL régularisée** : ajouter un terme de crête $\lambda I$
  à $S_T$, ce qui correspond à la régression $\ell_2$-régularisée, évitant
  les problèmes de singularité lorsque $d > N$.
- Cela ouvre la voie à une **ADL à noyau** en kernelisant le problème de
  régression plutôt que le critère de Fisher directement.


## Références

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 4.1.5. Springer.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Section 4.3. Springer.
3. Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification*, Chapitre 3. Wiley.
4. Paige, C. C. & Saunders, M. A. (1982). LSQR: An algorithm for sparse linear equations and sparse least squares. *ACM Transactions on Mathematical Software*, 8(1), 43–71.
5. Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(7), 711–720.
6. Friedman, J. H. (1989). Regularized discriminant analysis. *Journal of the American Statistical Association*, 84(405), 165–175.
7. Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589–609. -->
