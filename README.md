
# EEE3142-california-housing-GP
California housing price prediction using Gaussian Processes for Sogang Univ. EEE3142 (Introduction to Machine Learning)

## Highlights
* Gaussian process regression
* Elliptical slice sampling for sampling the hyperparameters
* GPU acceleration for Cholesky factorization

For more details, see `report/master.pdf`

## Results

5-fold cross-validation root-mean-squared error on the [California housing price dataset](https://www.kaggle.com/camnugent/california-housing-prices).

| Method | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|--------|-------:|-------:|-------:|-------:|-------:|--------:|
| Multi-layer Perceptron | 114,027 | 116,850 | 117,282 | 116,483 | 116,044 | 116,137 |
| Gaussian Process | **46,116** | **48,864** | **53,973** | **48,337** | **47,800** | **49,018** |
