# Importing the necessary libraries
import numpy as np


# define a function to calculate mean, median and standard deviation for 
# each perfomance metric
def calculate_mean_median_sd(model_data, model_name):
    for metric, values in model_data.items():
        mean = np.mean(values)
        median = np.median(values)
        sd = np.std(values)
        print(f"{model_name} ({metric}):")
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Standard Deviation: {sd:.2f}")
        print()




#------------Twitter US Airline Sentiment Dataset (Hyperparameter)------------#
# svm model
svm_model_twitter_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

# naive bayes model
nb_model_twitter_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

# random forest model
rf_model_twitter_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}





#------------IMDb Movie Review Dataset (Hyperparameter)------------#
# svm model
svm_model_imdb_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

# naive bayes model
nb_model_imdb_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

# random forest model
rf_model_imdb_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}





# Perform calculations for each model on the datasets
calculate_mean_median_sd(svm_model_twitter_h, "SVM (Twitter Hyperparameter)")
calculate_mean_median_sd(nb_model_twitter_h, "Naive Bayes (Twitter Hyperparameter)")
calculate_mean_median_sd(rf_model_twitter_h, "Random Forest (Twitter Hyperparameter)")
calculate_mean_median_sd(svm_model_imdb_h, "SVM (IMDb Hyperparameter)")
calculate_mean_median_sd(nb_model_imdb_h, "Naive Bayes (IMDb Hyperparameter)")
calculate_mean_median_sd(rf_model_imdb_h, "Random Forest (IMDb Hyperparameter)")