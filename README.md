# Ensemble:

Ensemble method is a machine learning technique that combines several base models in order to produce one optimal predictive model.


Ensemble learning, a method in machine learning, aims to improve forecast accuracy and resilience by combining predictions from various models. Its goal is to reduce errors or biases present in individual models by utilizing the collective intelligence of the ensemble.

# Why are Ensemble models developed over single traditional models?

a) Improved Predictive Performance: By combining multiple models, ensemble methods can reduce bias and variance, leading to more accurate predictions.

b) Reduction of Overfitting: By combining the predictions of multiple models, ensemble methods can mitigate the impact of individual model biases and errors, leading to more robust and generalized predictions, , especially when dealing with noisy or high-dimensional data.

c) Handling Complex Relationships: By combining multiple models with different perspectives or learning algorithms, ensemble methods can better capture diverse patterns and structures in the data, leading to improved performance. 

d) Less affected by outliers:  Ensemble models tend to be more robust and stable compared to single traditional models as they can handle variations in the data and mitigate the impact of outliers or noisy instances, resulting in more reliable predictions.

#  Some common Ensembled algorithms:

A) Bagging (Bootstrap Aggregating:parallel ensemble method): The predictions of individual models are then aggregated through averaging (for regression tasks) or voting (for classification tasks) to make the final prediction.
 Each instance is trained in parallel, meaning that they can be executed simultaneously on separate computational units (e.g., CPU cores, threads, or distributed computing nodes).

Example of parallel ensemble techniques: Bagging (Bootstrap Aggregating)

Bootstrap: Random sampling with replacement
  
In parallel ensemble techniques, base learners are generated in a parallel format, e.g., random forest. Parallel methods utilize the parallel generation of base learners to encourage independence between the base learners. The independence of base learners significantly reduces the error due to the application of averages.



Bagging is a parallel ensemble method where multiple base models are trained independently in parallel. Each base model is trained on a random subset of the training data (bootstrap sample), and their predictions are combined through averaging (for regression) or voting ( for classification) to make the final prediction.

Steps involved:
a) Bootstrap Sampling: Randomly select subsets of the training data with replacement (bootstrap samples).
b) Base Model Training: Train multiple base models (e.g., Decision Trees) independently on each bootstrap sample.
c) Prediction Combination: Combine the predictions from all base models through averaging (for regression) or voting (for classification) to make the final prediction.




 
Examples include Random Forest for decision trees and Bagged SVM for support vector machines.

B) Boosting (Sequential ensemble techniques):

Boosting aims to correct the errors of the previous learners and improve overall performance iteratively.
Examples include AdaBoost (Adaptive Boosting), Gradient Boosting Machines (GBM), XGBoost, and LightGBM.

A) Sequential ensemble techniques generate base learners in a sequence, e.g., Adaptive Boosting (AdaBoost). The sequential generation of base learners promotes the dependence between the base learners. The performance of the model is then improved by assigning higher weights to previously misrepresented learners.


Example of Sequential ensemble techniques: Boosting (e.g., AdaBoost, Gradient Boosting)
Boosting is a sequential ensemble method where base models are trained sequentially, and each subsequent model focuses on learning from the mistakes of the previous models. It iteratively adjusts the weights of training instances based on the performance of previous models.

Steps involved:
1. Base Model Training: Train a base model (e.g., Decision Tree) on the entire training data.
2. Weighted Prediction: Assign higher weights to misclassified instances and lower weights to correctly classified instances.
3. Base Model Iteration: Train the next base model, giving more emphasis to the previously misclassified instances.
4. Sequential Combination: Combine the predictions of all base models through weighted voting or averaging, where models with better performance have higher weights.






C) Stacking (Stacked Generalization): 

Stacking can involve multiple layers of base learners and meta-learners, creating a hierarchical ensemble.

Meta-learning algorithms typically involve training a model on a variety of different tasks, with the goal of learning generalizable
knowledge that can be transferred to new tasks.


Base-Models (Level-0 Models): Models that fit the training data and predict out-of-sample data.
Meta-Model (Level-1 Model): Model that fits on the prediction from base-models and learns how to best combine the
predictions on other tasks as well.

This is different from traditional machine learning, where a model is typically trained on a single task and then used for that task
alone.


# GIST: 
Ensemble learning refers to algorithms that combine the predictions from two or more models.
The idea of ensemble methods is to try reducing bias and/or variance of such weak learners by combining several of them
together to create a strong learner (or ensemble model) that achieves better performances. We call weak learners (or base
models) models that can be used as building blocks for designing more complex models by combining several of them. Weak
Learners: A ‘weak learner’ is any ML algorithm (for regression/classification) that provides an accuracy slightly better than
random guessing.

The three main classes of ensemble learning methods are bagging, boosting, and stacking.
A) Bagging ( Parallel ensemble technique) involves fitting many decision trees on different samples of the same dataset and averaging the predictions.

B) Boosting (sequential ensemble technique) involves adding ensemble members sequentially that correct the predictions made by prior models and outputs a
weighted average of the predictions.

Stacking creates a meta-model.

# CONCLUSION:

If you want to reduce the overfitting or variance of your model, you use bagging.

If you are looking to reduce underfitting or bias, you use boosting.

However, if you want to increase predictive accuracy, use stacking.

Bagging and boosting both work with homogeneous weak learners.

Stacking works using heterogeneous solid learners.

All three of these methods can work with either classification or regression problems.








