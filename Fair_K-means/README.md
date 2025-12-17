# Fair K-Means++ (Force-Based Assignment)

This repository implements **Fair K-Means++**, a fairness-aware extension of the classical K-Means algorithm, inspired by a **physics-based force formulation**.

## Overview

Standard K-Means clusters data solely based on geometric proximity, ignoring sensitive attributes. This may lead to highly imbalanced clusters.
Fair K-Means++ introduces **fairness-aware forces** that guide point-to-cluster assignments toward balanced representations of a binary sensitive attribute (e.g., gender, race).

The algorithm preserves the centroid-based structure of K-Means while modifying the assignment rule.

---

## Standard K-Means Objective

Given data points ( X = {x_1,\dots,x_n} \subset \mathbb{R}^d ), K-Means minimizes:
[
\min_{C} \sum_{j=1}^{k} \sum_{x_i \in C_j} |x_i - \mu_j|_2^2
]

Assignments are traditionally made by nearest centroid.

---

## Fairness Definition

Each point has a **binary sensitive attribute** (red / blue).

For cluster ( C_j ):

* ( C_j^r ): red points
* ( C_j^b ): blue points

Cluster balance is defined as:
[
\text{bal}(C_j) = \min\left(\frac{|C_j^r|}{|C_j^b|}, \frac{|C_j^b|}{|C_j^r|}\right) \in [0,1]
]

Cluster imbalance:
[
\text{imb}(C_j) = 1 - \text{bal}(C_j)
]

---

## Physics-Inspired Interpretation

We reinterpret cluster assignment as a **force maximization problem**.

### Geometric Force

Standard K-Means assignment:
[
\arg\min_j |x_i - \mu_j|^2
\quad \Longleftrightarrow \quad
\arg\max_j \frac{1}{|x_i - \mu_j|}
]

### Fairness via Charges

* Each point is assigned a charge:

  * red → ( q_x = +1 )
  * blue → ( q_x = -1 )
* Each centroid receives a charge:
  [
  q_{\mu_j} = \text{imb}(C_j) \cdot \text{sign}(\text{majority color})
  ]

---

## Combined Force

The total force exerted by centroid ( \mu_j ) on point ( x_i ) is:
[
F(x_i, \mu_j)
=============

\Big((1-\lambda) - \lambda q_x q_{\mu_j}\Big)
\cdot
\frac{1}{|x_i - \mu_j|}
]

Each point is assigned to the centroid that **maximizes** this force:
[
\ell_i = \arg\max_j F(x_i, \mu_j)
]

* ( \lambda = 0 ): standard K-Means
* Larger ( \lambda ): stronger fairness influence

To ensure non-negative forces:
[
0 \le \lambda \le 0.5
]

---

## Algorithm

1. Initialize centroids (random or K-Means++).
2. Assign each point by maximizing total force.
3. Update centroids as cluster means.
4. Recompute imbalance and charges.
5. Repeat until convergence.

Multiple runs are supported to reduce sensitivity to initialization.

---

## Features

* K-Means++ initialization
* Force-based fair assignment
* Explicit fairness–geometry trade-off
* Multiple initialization strategies
* SSE and fairness evaluation metrics

---

## Usage

```python
model = KMeansBalanced_pp(
    n_clusters=3,
    lambda_=0.3,
    init_mode="kmeans++"
)
model.fit(X, attributes)
labels = model.labels_
```

---

## Dependencies

* numpy
* scikit-learn

---

## Notes

This method bridges centroid-based clustering with fairness constraints using a simple, interpretable force-based formulation.
