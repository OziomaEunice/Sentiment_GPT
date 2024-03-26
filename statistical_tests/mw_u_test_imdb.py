""" 
This file focuses on using a statistical significance test (SST) to determine
whether the mean of two samples is significantly different from each other.
For this case, the naive bayes, svm, and random forest are compared to each other.
The SST that is used is the Mann-Whitney U test. 
"""


# Importing the necessary libraries
from scipy.stats import mannwhitneyu

#------------IMDb Movie Review Dataset------------#
# Accuracy
svm_model_twitter_nh_a = [] # svm model accuracy (no hyperparameters)
svm_model_twitter_h_a = [] # svm model accuracy (hyperparameters)
nb_model_twitter_nh_a = [] # naive bayes model accuracy (no hyperparameters)
nb_model_twitter_h_a = [] # naive bayes model accuracy (hyperparameters)
rf_model_twitter_nh_a = [] # random forest model accuracy (no hyperparameters)
rf_model_twitter_h_a = [] # random forest model accuracy (hyperparameters)

# Precision
svm_model_twitter_nh_p = [] # svm model precision (no hyperparameters)
svm_model_twitter_h_p = [] # svm model precision (hyperparameters)
nb_model_twitter_nh_p = [] # naive bayes model precision (no hyperparameters)
nb_model_twitter_h_p = [] # naive bayes model precision (hyperparameters)
rf_model_twitter_nh_p = [] # random forest model precision (no hyperparameters)
rf_model_twitter_h_p = [] # random forest model precision (hyperparameters)

# Recall
svm_model_twitter_nh_r = [] # svm model recall (no hyperparameters)
svm_model_twitter_h_r = [] # svm model recall (hyperparameters)
nb_model_twitter_nh_r = [] # naive bayes model recall (no hyperparameters)
nb_model_twitter_h_r = [] # naive bayes model recall (hyperparameters)
rf_model_twitter_nh_r = [] # random forest model recall (no hyperparameters)
rf_model_twitter_h_r = [] # random forest model recall (hyperparameters)

# F1-score
svm_model_twitter_nh_f1 = [] # svm model f1 (no hyperparameters)
svm_model_twitter_h_f1 = [] # svm model f1 (hyperparameters)
nb_model_twitter_nh_f1 = [] # naive bayes model f1 (no hyperparameters)
nb_model_twitter_h_f1 = [] # naive bayes model f1 (hyperparameters)
rf_model_twitter_nh_f1 = [] # random forest model f1 (no hyperparameters)
rf_model_twitter_h_f1 = [] # random forest model f1 (hyperparameters)





#------------Calculate Mann-Whitney U test for IMDb Movie Review Dataset------------#