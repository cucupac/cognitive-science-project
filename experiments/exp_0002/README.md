# Experiment 2: SVM Classifier

Mirrors Experiment 1 but uses SVM with RBF kernel instead of logistic regression.

### Pipeline

1. Uses same precomputed CLIP embeddings from Experiment 1
2. Classifies with SVM (RBF kernel) using 5-fold stratified cross-validation
3. Evaluates with the same metrics

### Output

-   Accuracy: `experiments/exp_0002/results/data/combined/results_svm.csv`
-   Visualizations: `experiments/exp_0002/results/images/combined/scatter_plots/`
