# Cracking the Beer Code: Discovering Flavor Groups with Clustering

An unsupervised learning analysis exploring whether beers form meaningful, natural
groupings based on their flavor and tasting characteristics — rather than their
traditionally labeled style — using t-SNE, DBSCAN, and fuzzy c-means clustering.

## Overview

Beer style labels (IPA, Stout, Lager, etc.) are useful but often don't capture how
similar two beers actually *taste*. This project asks: if you ignore style labels
entirely and cluster beers purely on their flavor profile (bitterness, sweetness,
sourness, maltiness, hoppiness, ABV, and more), what natural groupings emerge?

The analysis walks through the full unsupervised learning pipeline:
- Data cleaning and outlier detection (kNN-distance analysis)
- Clusterability assessment via t-SNE across multiple perplexity values
- DBSCAN, evaluated across a parameter grid, plus a noise-filtered k-means follow-up
- Fuzzy c-means, tuned across cluster count and fuzziness parameter
- Cross-algorithm comparison (Adjusted Rand Index) and cluster separation/sparsity analysis

**Key finding:** Beer flavor profiles form a continuous, overlapping spectrum rather
than cleanly separated groups — DBSCAN in particular struggles to find any clean
density-based structure, which is itself informative. A fuzzy, partition-based
approach (fuzzy c-means, 5 clusters) produces the most usable result, recovering five
recognizable flavor groups (Sweet/Malty, Sour, Malty, mixed-Malty, Hoppy/Malty) while
still representing how much beers straddle multiple categories.

## Author

This started as a group final project for **STAT 437 (Unsupervised Learning)** at the
University of Illinois Urbana-Champaign, with three contributors: Jason Ye, Kenshi
King, and Yuda Zhu. **[Kenshi King]** was the primary contributor (~80% of the analysis
and write-up) and independently revised and expanded the report afterward — addressing
instructor feedback and adding additional analysis — for use as a graduate school
application portfolio piece.

## Dataset

- **Source:** [Beer Profile and Ratings Data Set](https://www.kaggle.com/datasets/ruthgn/beer-profile-and-ratings-data-set) (Kaggle)
- ~3,200 unique beers across ~900 breweries, combining tasting/flavor profiles with
  consumer review data
- The CSV is not included in this repo (see `.gitignore`) — download it directly from
  the Kaggle link above and place it in the project root as `beer_profile_and_ratings.csv`

## Repository Contents

```
.
├── beer_analysis.ipynb        # Full analysis notebook (report + code)
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## Running It Yourself

The notebook renders directly on GitHub with all plots and outputs already saved, so
you don't need to run anything just to read it. If you'd like to re-run or modify the
analysis:

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download beer_profile_and_ratings.csv from the Kaggle link above
#    and place it in this folder

# 5. Launch Jupyter
jupyter notebook beer_analysis.ipynb
```

## Methods Used

- **Preprocessing:** missing value handling, outlier/noise detection via kNN-distance
  plots, feature scaling (StandardScaler)
- **Clusterability assessment:** t-SNE across 6 perplexity values × 2 random states
- **Algorithm 1 — DBSCAN:** grid search over `eps`/`minPts`, cluster-sorted similarity
  matrices, noise-filtered k-means as a follow-up
- **Algorithm 2 — Fuzzy C-Means:** tuned over cluster count and fuzziness parameter,
  evaluated via NDPC, membership-confidence distributions, and silhouette score
- **Comparison:** Adjusted Rand Index between the two final clusterings; within-cluster
  sparsity and between-cluster separation using Euclidean distance in scaled feature space

## References

Full citations are listed at the end of the notebook, including:
- Schreurs et al. (2024), *Nature Communications* — machine learning for beer flavor
- Myles et al. (2022), *Nutrients* — craft/low-alcohol beer characteristics
- Richter (2023), *Medical News Today* — beer ABV context
- Kaggle dataset source (ruthgn, 2021)
