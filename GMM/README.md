Gaussian Mixture Model from scratch:
A clean NumPy implementation of a Gaussian Mixture Model (GMM) fitted with the Expectation-Maximization (EM) algorithm. No sklearn.mixture used.
What it does

Fits K Gaussian components to unlabelled data
E-step: computes soft responsibilities per component
M-step: updates weights, means, and covariance matrices
K-Means++ initialisation for better convergence
Convergence check via log-likelihood

Usage
bashpip install numpy matplotlib
python gmm.py
This runs a demo on synthetic 2D data from 3 Gaussians and saves a plot to gmm_result.png.