# ML4Sci-prep
# End-to-End Calorimeter Shower Classification: GSoC 2026 PoC

## Overview
This repository contains a Proof of Concept (PoC) developed for my **Google Summer of Code (GSoC) 2026** application with **Machine Learning for Science (ML4SCI)**. 

My GSoC proposal, **"ML-Based Simulation Bias Analysis: Quantifying Pythia-Herwig Discrepancies,"** focuses on automating the extraction of systematic biases between Monte Carlo event generators and real detector data. This notebook demonstrates the foundational architecture required for this task by building an End-to-End (E2E) Deep Learning pipeline that classifies synthetic calorimeter showers while providing direct physical interpretability.

## Architecture & Features
This pipeline is built using **PyTorch** and is optimized for High-Energy Physics (HEP) MLOps:
* **Robust CNN Architecture:** Utilizes a Convolutional Neural Network equipped with `BatchNorm2d` and Spatial Dropout to prevent overfitting on sparse physics arrays.
* **Physics Metrics:** Evaluates model convergence using the `roc_auc_score`, the standard metric for binary classification in HEP.
* **SHAP Interpretability:** Implements the `shap.GradientExplainer` to extract feature importance. The model does not act as a black box; it outputs explicit heatmaps showing exactly which kinematic regions drove the Pythia vs. Herwig classification bias.

## How to Run
1. Open the notebook in Google Colab (Recommended for GPU acceleration).
2. Ensure the runtime is set to **T4 GPU**.
3. Install the required interpretability dependencies: `pip install torch torchvision shap scikit-learn`
4. Run all cells. The script will automatically generate synthetic 2D shower data, train the classifier, and output the SHAP interpretability grids.

## About the Author
**Mithilesh A** Software Developer specializing in Deep Learning, complex systems architecture, and physics-informed neural networks. 
* **GitHub:** [https://github.com/Mithil-7](https://github.com/Mithil-7)
* **LinkedIn:** [https://www.linkedin.com/in/mithilesh-a-07486935b/](https://www.linkedin.com/in/mithilesh-a-07486935b/)
