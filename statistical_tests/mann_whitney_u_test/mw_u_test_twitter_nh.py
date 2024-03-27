""" 
This file focuses on using a statistical significance test (SST) to determine
whether the mean of two samples is significantly different from each other.
For this case, the naive bayes, svm, and random forest are compared to each other.
The SST that is used is the Mann-Whitney U test. 
"""


# Importing the necessary libraries
from scipy.stats import mannwhitneyu

#------------Twitter US Airline Sentiment Dataset (No Hyperparameter)------------#
# svm model
svm_model_twitter_nh = {
    "accuracy": [76.82,76.82,76.82,76.82,76.82,76.82,76.82,76.82,76.82,76.82,
                 76.82,76.82,76.82,76.82,76.82,76.82,76.82,76.82,76.82,76.82],
    "precision": [0.73,0.73,0.73,0.73,0.73,0.73,0.73,0.73,0.73,0.73,
                  0.73,0.73,0.73,0.73,0.73,0.73,0.73,0.73,0.73,0.73],
    "recall": [0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,
               0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66],
    "f1_score": [0.68,0.68,0.68,0.68,0.68,0.68,0.68,0.68,0.68,0.68,
                 0.68,0.68,0.68,0.68,0.68,0.68,0.68,0.68,0.68,0.68]
}

# naive bayes model
nb_model_twitter_nh = {
    "accuracy": [67.62,67.62,67.62,67.62,67.62,67.62,67.62,67.62,67.62,67.62,
                 67.62,67.62,67.62,67.62,67.62,67.62,67.62,67.62,67.62,67.62],
    "precision": [0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,
                  0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78,0.78],
    "recall": [0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,
               0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43],
    "f1_score": [0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,
                 0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43,0.43]
}

# random forest model
rf_model_twitter_nh = {
    "accuracy": [75.30,75.04,75.48,75.25,75.30,75.10,75.25,75.52,75.68,75.63,
                 75.16,75.03,75.26,75.31,75.15,75.30,75.23,75.51,75.04,75.67],
    "precision": [0.71,0.72,0.71,0.71,0.71,0.70,0.71,0.71,0.72,0.71,
                  0.71,0.70,0.71,0.71,0.71,0.71,0.71,0.71,0.70,0.71],
    "recall": [0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.64,0.64,
               0.63,0.63,0.63,0.64,0.63,0.63,0.63,0.64,0.63,0.64],
    "f1_score": [0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.67,
                 0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66]
}


# Define a function that will calculate the Mann-Whitney U test
# and prints the results for each score performance metric
def calculate_mannwhitneyu(model_data1, model_data2, model_name1, model_name2, metrics):
    for metric in metrics:
        statistic, p_value = mannwhitneyu(model_data1[metric], model_data2[metric])
        print(f"{model_name1} vs {model_name2} ({metric}):")
        print("Mann-Whitney U Statistic:", statistic)
        print("p-value:", p_value)
        if p_value < 0.05:
            print(f"There is a statistically significant difference between {model_name1} and {model_name2} for {metric}.")
        else:
            print(f"There is no statistically significant difference between {model_name1} and {model_name2} for {metric}.")
        print()

# Specify performance metrics to evaluate
metrics_to_evaluate = ["accuracy", "precision", "recall", "f1_score"]

# Perform Mann-Whitney U test for different model comparisons
calculate_mannwhitneyu(svm_model_twitter_nh, nb_model_twitter_nh, "SVM", "Naive Bayes", metrics_to_evaluate) # svm no-hyperparameters vs nb no-hyperparameters
calculate_mannwhitneyu(svm_model_twitter_nh, rf_model_twitter_nh, "SVM", "Random Forest", metrics_to_evaluate) # svm no-hyperparameters vs rf no-hyperparameters
calculate_mannwhitneyu(nb_model_twitter_nh, rf_model_twitter_nh, "Naive Bayes", "Random Forest", metrics_to_evaluate) # nb no-hyperparameters vs rf no-hyperparameters