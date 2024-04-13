# Binary Classification of Neutrino Interactions
This repository contains the code and report for the Neutrino Interactions Project completed in 2024 for the Practical Machine Learning for Physicists course at UCL.

Below is a brief summary of the [project report](https://github.com/Sim-Ood/Binary-Classification-of-Neutrino-Interactions/blob/main/Classifying%20Neutrinos%20Report.pdf)

## Introduction

The aim of this project is to train a machine learning classifier on simulated images resembling the particle tracks made in the NOvA detector, to identify muon neutrino charged-current (CC) events. The secondary aim is to investigate the relationship between the image metadata, such as neutrino energy, and the performance of the classifier. There were 200 available files of data for this project which contained images of simulated neutrino interactions and their associated metadata.

## Preprocessing Data

There are 17 classes of interaction, 4 of which are $\nu_{\mu}$ CC. By relabelling $\nu_{\mu}$ CC type events as 1 and all other classes as 0, the data is prepared for binary classification where 1 is positive and 0 is negative. The dataset is imbalanced with 88.24% of samples in the positive class, and 11.76% of samples in the negative class. Consequently, if the model learnt to label all samples as positive instead of distinguishing between positive and negative results, it could still achieve a misleading $\sim$88% accuracy. To address this the following strategies are considered.

- Resample the dataset to overrepresent the negative class and balance the distribution of samples over the two classes.
- Implement class weights in the loss function to increase the penalty for incorrectly classified negative samples.
- Evaluate model performance on multiple metrics alongside accuracy.
- Identify the optimal threshold at which the model's probability scores is classed as positive and recalculate the accuracy. 



Hence, the model's ability to predict on the negative minority class is a better indicator of its performance. 
