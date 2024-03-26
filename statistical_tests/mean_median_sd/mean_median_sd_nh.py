# Importing the necessary libraries
import numpy as np


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
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

# random forest model
rf_model_twitter_nh = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}





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
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}

# random forest model
rf_model_imdb_nh = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}




# define a function to calculate mean, median and standard deviation for 
# each perfomance metric
def calculate_mean_median_sd(model_data, model_name):
    for metric, values in model_data.items():
        mean = np.mean(values)
        median = np.median(values)
        sd = np.std(values)

        # printing outputs
        print(f"{model_name} ({metric}):")
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Standard Deviation: {sd:.2f}")
        print()


# Perform calculations for each model on the datasets
calculate_mean_median_sd(svm_model_twitter_nh, "SVM (Twitter No Hyperparameter)")
calculate_mean_median_sd(nb_model_twitter_nh, "Naive Bayes (Twitter No Hyperparameter)")
calculate_mean_median_sd(rf_model_twitter_nh, "Random Forest (Twitter No Hyperparameter)")
calculate_mean_median_sd(svm_model_imdb_nh, "SVM (IMDb No Hyperparameter)")
calculate_mean_median_sd(nb_model_imdb_nh, "Naive Bayes (IMDb No Hyperparameter)")
calculate_mean_median_sd(rf_model_imdb_nh, "Random Forest (IMDb No Hyperparameter)")