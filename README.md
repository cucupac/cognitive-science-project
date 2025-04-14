# Is a Picture Worth a Thousand Words?

**Overview:**
This repository contains the core codebase for a cognitive science project investigating how text descriptions can “rescue” classification performance when images are degraded. We leverage OpenAI’s CLIP embeddings to fuse textual and visual information under varying conditions of image corruption (pixel dropout) and text detail.

**Key Features:**

1. **Data Preprocessing:** Scripts to handle the dogs vs. cats dataset, apply controlled pixel dropout, and prepare text descriptions of different levels of detail.
2. **Embedding Generation:** Code to embed both images and text into CLIP’s shared semantic space.
3. **Multimodal Fusion:** Multiple strategies to combine text and image embeddings via a weighted parameter (α), including purely visual, purely textual, and hybrid modes.
4. **Experimentation:** Automated pipelines for training logistic regression and SVM classifiers, running k-fold cross-validations, and generating results for different dropout levels.
5. **Visualization:** Scripts to produce and save plots that illustrate “rescue effects” and classification trends across a range of modality weightings.

**How to Use:**

1. **Clone This Repository:**  
   `git clone https://github.com/cucupac/cognitive-science-project.git`
2. **Install Dependencies:**  
   Make sure you have Python 3.7+ and install required libraries via `pip install -r requirements.txt`.
3. **Run the Experiments:**  
   Execute scripts inside the `experiments/` folder to reproduce key results and figures.
4. **View Results:**  
   Plots, metrics, and logs will be saved in the designated `results/` directory.

**Report and Further Details:**
A detailed write-up of the methodology, rationale, and findings—including references to CLIP and the pixel-dropout procedure—can be found in the Overleaf report:
[https://www.overleaf.com/read/zfjmpzffpxwk#4b86b9](https://www.overleaf.com/read/zfjmpzffpxwk#4b86b9)

**License & Acknowledgments:**

-   This project uses OpenAI’s CLIP for multimodal embedding.
-   Dataset: [Kaggle: Dogs vs. Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
-   Code and instructions released under an MIT License (see repository for details).
