""" 
This file focuses on using a statistical significance test (SST) to determine
whether the mean of two samples is significantly different from each other.
For this case, the naive bayes, svm, and random forest are compared to each other.
The SST that is used is the Mann-Whitney U test. 
"""


# Importing the necessary libraries
from scipy.stats import mannwhitneyu

#------------IMDb Movie Review Dataset (No Hyperparameter)------------#
# svm model
svm_model_imdb_nh = {
    "accuracy": [57.36,57.36,57.36,57.36,57.36,57.36,57.36,57.36,57.36,57.36,
                 57.36,57.36,57.36,57.36,57.36,57.36,57.36,57.36,57.36,57.36],
    "precision": [0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,
                  0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49],
    "recall": [0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,
               0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72],
    "f1_score": [0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,
                 0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33]
}

# naive bayes model
nb_model_imdb_nh = {
    "accuracy": [50.43,50.43,50.43,50.43,50.43,50.43,50.43,50.43,50.43,50.43,
                 50.43,50.43,50.43,50.43,50.43,50.43,50.43,50.43,50.43,50.43],
    "precision": [0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,
                  0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49,0.49],
    "recall": [0.67,0.67,0.67,0.67,0.67,0.67,0.67,0.67,0.67,0.67,
               0.67,0.67,0.67,0.67,0.67,0.67,0.67,0.67,0.67,0.67],
    "f1_score": [0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,
                 0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.23]
}

# random forest model
rf_model_imdb_nh = {
    "accuracy": [48.08,47.90,48.01,47.01,48.30,49.25,48.02,50.52,49.01,49.50,
                 46.98,49.06,50.61,46.80,49.13,47.38,49.36,47.46,47.24,49.35],
    "precision": [0.45,0.45,0.46,0.46,0.45,0.45,0.46,0.46,0.45,0.46,
                  0.46,0.45,0.45,0.45,0.46,0.46,0.46,0.46,0.46,0.46],
    "recall": [0.65,0.65,0.65,0.65,0.66,0.66,0.65,0.67,0.66,0.66,
               0.65,0.66,0.67,0.65,0.66,0.65,0.66,0.65,0.65,0.66],
    "f1_score": [0.33,0.32,0.33,0.31,0.32,0.33,0.32,0.32,0.32,0.32,
                 0.32,0.32,0.33,0.32,0.32,0.32,0.33,0.32,0.32,0.33]
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
calculate_mannwhitneyu(svm_model_imdb_nh, nb_model_imdb_nh, "SVM", "Naive Bayes", metrics_to_evaluate) # svm no-hyperparameters vs nb no-hyperparameters
calculate_mannwhitneyu(svm_model_imdb_nh, rf_model_imdb_nh, "SVM", "Random Forest", metrics_to_evaluate) # svm no-hyperparameters vs rf no-hyperparameters
calculate_mannwhitneyu(nb_model_imdb_nh, rf_model_imdb_nh, "Naive Bayes", "Random Forest", metrics_to_evaluate) # nb no-hyperparameters vs rf no-hyperparameters