""" 
This file focuses on using a statistical significance test (SST) to determine
whether the mean of two samples is significantly different from each other.
For this case, the naive bayes, svm, and random forest are compared to each other.
The SST that is used is the Mann-Whitney U test. 
"""


# Importing the necessary libraries
from scipy.stats import mannwhitneyu

#------------IMDb Movie Review Dataset (Hyperparameter)------------#
# svm model
svm_model_imdb_h = {
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
nb_model_imdb_h = {
    "accuracy": [74.96, 74.96,74.96,74.96,74.96,74.96,74.96,74.96,74.96,74.96,
                 74.96,74.96,74.96,74.96,74.96,74.96,74.96,74.96,74.96,74.96],
    "precision": [0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,
                  0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72],
    "recall": [0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,
               0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60],
    "f1_score": [0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.63,
                 0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.63,0.63]
}

# random forest model
rf_model_imdb_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
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
calculate_mannwhitneyu(svm_model_imdb_h, nb_model_imdb_h, "SVM", "Naive Bayes", metrics_to_evaluate) # svm hyperparameters vs nb hyperparameters
calculate_mannwhitneyu(svm_model_imdb_h, rf_model_imdb_h, "SVM", "Random Forest", metrics_to_evaluate) # svm hyperparameters vs rf hyperparameters
calculate_mannwhitneyu(nb_model_imdb_h, rf_model_imdb_h, "Naive Bayes", "Random Forest", metrics_to_evaluate) # nb hyperparameters vs rf hyperparameters