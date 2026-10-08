# Machine Learning Notes

## Introduction
Machine learning is a field of artificial intelligence that gives computers the ability to learn patterns from data without being explicitly programmed. A model is trained on a training set and evaluated on a separate test set to estimate how well it generalizes to unseen data.

## Supervised Learning
Supervised learning is a type of machine learning where the model learns from labeled examples, meaning every input comes with the correct output. Common supervised tasks are classification, where the output is a category, and regression, where the output is a continuous number. Examples of supervised algorithms include linear regression, logistic regression, decision trees, support vector machines and neural networks.

## Unsupervised Learning
Unsupervised learning is a type of machine learning where the model finds structure in unlabeled data. Clustering groups similar data points together, for example with k-means. Dimensionality reduction, such as principal component analysis (PCA), compresses data into fewer features while keeping most of the variance. Unlike supervised learning, there is no ground-truth label to compare predictions against.

## Overfitting and Underfitting
Overfitting is when a model learns the noise and random fluctuations in the training data instead of the underlying pattern. An overfit model has very low training error but high test error, so it generalizes poorly. Underfitting is when a model is too simple to capture the underlying pattern, giving high error on both training and test data.

Overfitting can be reduced with more training data, regularization, early stopping, dropout, simpler models and cross-validation. Regularization is a technique that adds a penalty on model complexity to the loss function, for example the L2 penalty in ridge regression or the L1 penalty in lasso.

## Bias-Variance Trade-off
Bias is the error caused by overly simple assumptions in the model. Variance is the error caused by sensitivity to small fluctuations in the training set. High bias leads to underfitting and high variance leads to overfitting; a good model balances the two.

## Cross-Validation
Cross-validation is a resampling technique used to estimate model performance on unseen data. In k-fold cross-validation the data is split into k folds; the model is trained on k-1 folds and validated on the remaining fold, and this is repeated k times.

## Transfer Learning
Transfer learning is a technique where a model trained on one large task is reused as the starting point for a different but related task. For example, a convolutional neural network pretrained on ImageNet can be fine-tuned on a small medical imaging dataset. Transfer learning reduces the amount of labeled data and compute needed and usually improves accuracy when the target dataset is small.

## Gradient Descent
Gradient descent is an optimization algorithm that iteratively updates model parameters in the direction of the negative gradient of the loss function. The learning rate controls the step size: too large and training diverges, too small and training is slow. Stochastic gradient descent uses a single example or a mini-batch per update.
