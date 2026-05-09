# Model Comparison Report

## Classification

| model         |   accuracy |   precision_macro |   recall_macro |   f1_macro |   f1_weighted |   train_time_s |   predict_time_s |
|:--------------|-----------:|------------------:|---------------:|-----------:|--------------:|---------------:|-----------------:|
| knn           |     0.5159 |            0.5132 |         0.4919 |     0.4938 |        0.5006 |           0    |             2.83 |
| svm           |     0.5137 |            0.5438 |         0.4753 |     0.4963 |        0.5115 |          41.63 |             4.25 |
| decision_tree |     0.3346 |            0.3224 |         0.3025 |     0.3099 |        0.3339 |           2.82 |             0    |
| bagging_dt    |     0.4588 |            0.5337 |         0.4106 |     0.4332 |        0.443  |           7.74 |             0.25 |
| adaboost_dt   |     0.3977 |            0.4056 |         0.3217 |     0.3247 |        0.3884 |         138.68 |             0.05 |


## Regression (valence prediction)

| model                 |   rmse |    mae |     r2 |   train_time_s |
|:----------------------|-------:|-------:|-------:|---------------:|
| linear_regression     | 0.5819 | 0.4988 | 0.2105 |           0.04 |
| polynomial_regression | 0.5622 | 0.4523 | 0.2629 |           1.81 |


## Clustering

| model   |   silhouette |   davies_bouldin |   adjusted_rand_index |   train_time_s |
|:--------|-------------:|-----------------:|----------------------:|---------------:|
| kmeans  |       0.0191 |           4.2037 |                0.0104 |           0.52 |


## Reinforcement Learning

- episodes: **2000**
- final avg reward (last 100): **17.670**
- first avg reward (first 100): **-20.571**
- max reward: **41.500**
- train time: **0.27s**
