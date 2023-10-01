# Review: Uncertainty Quantification in Deep Learning

This repository is a review of different methods to estimate uncertainty in deep learning models.

## Table of Contents
- [Review: Uncertainty Quantification in Deep Learning](#review-uncertainty-quantification-in-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [1. Context](#1-context)
  - [2. Bayesian Neural Network](#2-bayesian-neural-network)
  - [3. Bayes by Backprop](#3-bayes-by-backprop)
  - [4. Monte-Carlo Dropout as a Bayesian Approximation](#4-monte-carlo-dropout-as-a-bayesian-approximation)
  - [5. Deep Ensembles](#5-deep-ensembles)
  - [6. Aleatoric vs. Epistemic Uncertainty](#6-aleatoric-vs-epistemic-uncertainty)

## 1. Context
The goal of this review is to better understand the different methods to estimate uncertainty in deep learning models. This is a fundamental problem in machine learning, especially in safety-critical applications. For example, in autonomous driving, it is important to know when the model is uncertain about its prediction. This is because the model should not be trusted when it is uncertain. In this case, the model should defer to a human driver.

*Note: All proofs and explanations are given in the referenced papers.*

## 2. Bayesian Neural Network
The referenced paper contains a good introduction to Bayesian Neural Networks (BNNs). The main idea is to treat the weights of a neural network as random variables and to infer their posterior distribution given the data. This is done by using Bayes' rule:
$$  p(w|D) = \frac{p(D|w)p(w)}{p(D)} $$
where $w$ are the weights of the neural network, $D$ is the data, $p(w|D)$ is the posterior distribution of the weights given the data, $p(D|w)$ is the likelihood of the data given the weights, $p(w)$ is the prior distribution of the weights, and $p(D)$ is the marginal likelihood of the data.

The covered methods are all included in the category of Variational Inference (VI). VI is a method to approximate the posterior distribution of the weights. The main idea is to find a distribution $q(w)$ that is close to the true posterior $p(w|D)$. This is done by minimizing the Kullback-Leibler (KL) divergence between $q(w)$ and $p(w|D)$:
$$ \min_{q(w)} KL(q(w)||p(w|D)) = \min_{q(w)} \int q(w) \log \frac{q(w)}{p(w|D)} dw $$

This integral is intractable and therefore, we use methods to approximate.



References:
- [Hands-on Bayesian Neural Networks - A Tutorial for Deep Learning Users](https://arxiv.org/pdf/2007.06823.pdf#:~:text=A%20BNN%20is%20defined%20slightly,y%20%3D%20%CE%A6(x).)

## 3. Bayes by Backprop
The main idea behind Bayes by Backprop is to use the reparameterization trick to approximate the posterior distribution of the weights.
References:
- [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1505.05424.pdf)

## 4. Monte-Carlo Dropout as a Bayesian Approximation
The main idea behind Monte-Carlo Dropout is to use dropout at test time to estimate the uncertainty. This is done by sampling the weights of the neural network using dropout and by computing the variance of the predictions of the network. 

References:
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142v6.pdf)
- [Dropout as a Bayesian Approximation: Appendix](https://arxiv.org/pdf/1506.02157.pdf)

## 5. Deep Ensembles
The main idea behind Deep Ensembles is to train multiple neural networks with different initializations and to use the ensemble of these networks to estimate the uncertainty. The uncertainty is estimated by computing the variance of the predictions of the ensemble.

References:
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/pdf/1612.01474.pdf)

## 6. Aleatoric vs. Epistemic Uncertainty
The main idea behind Aleatoric vs. Epistemic Uncertainty is to decompose the uncertainty into two components: aleatoric and epistemic. Aleatoric uncertainty is the uncertainty due to the noise in the data. Epistemic uncertainty is the uncertainty due to the lack of data. 

Let's assume we have a model with uncertainty that outputs two quantities: the mean $\mu_i(x)$ and variance $\sigma^2_i(x)$, where $i\in[1,M]$ is an index for different samples or ensembles. Therefore, we have:
$$ \mu_*(x) = M^{-1} \sum_i \mu_i(x) $$
$$ \sigma^2_* = M^{-1} \sum_i (\sigma^2_i(x) + \mu^2_i(x)) - \mu^2_*(x) $$

We can rewrite the last expression as:
$$ \sigma^2_* = \mathbb{E}_i[\sigma^2_i(x)] + \mathbb{V}_i[\mu_i(x)] $$

The first term represents aleatoric uncertainty and the second epistemic uncertainty.

References:
- [A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement](https://arxiv.org/pdf/2204.09308.pdf)