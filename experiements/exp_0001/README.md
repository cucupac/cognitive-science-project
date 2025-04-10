# Experiment 1: Logistic Regression Classifier

Evaluates how combined CLIP image-text embeddings perform in a binary classification task using logistic regression.

### Pipeline

1. **Embeddings Generation**:

-   High/low-info images (via pixel dropout)
-   High/low-info text descriptions (varying detail)

2. **Embedding Combination**:

```
combined_vector = α·image_embedding + (1-α)·text_embedding,

Where α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
```

3. **Classification**: Logistic regression with 5-fold cross-validation

4. **Evaluation**:

-   Accuracy metrics saved to CSV
-   t-SNE visualizations of embedding space

### Output

-   Accuracy: `experiements/exp_0001/results/data/combined/results.csv`
-   Visualizations: `experiements/exp_0001/results/images/combined/scatter_plots/`
