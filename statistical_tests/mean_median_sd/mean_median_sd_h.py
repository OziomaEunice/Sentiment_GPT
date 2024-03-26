# Importing the necessary libraries
import numpy as np


#------------Twitter US Airline Sentiment Dataset (Hyperparameter)------------#
# svm model
svm_model_twitter_h = {
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
nb_model_twitter_h = {
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
rf_model_twitter_h = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}





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
    "accuracy": [57.41,57.41,57.41,57.41,57.41,57.41,57.41,57.41,57.41,57.41,
                 57.41,57.41,57.41,57.41,57.41,57.41,57.41,57.41,57.41,57.41],
    "precision": [0.47,0.47,0.47,0.47,0.47,0.47,0.47,0.47,0.47,0.47,
                  0.47,0.47,0.47,0.47,0.47,0.47,0.47,0.47,0.47,0.47],
    "recall": [0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,
               0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72],
    "f1_score": [0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,
                 0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34]
}

# random forest model
rf_model_imdb_h = {
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
calculate_mean_median_sd(svm_model_twitter_h, "SVM (Twitter Hyperparameter)")
calculate_mean_median_sd(nb_model_twitter_h, "Naive Bayes (Twitter Hyperparameter)")
calculate_mean_median_sd(rf_model_twitter_h, "Random Forest (Twitter Hyperparameter)")
calculate_mean_median_sd(svm_model_imdb_h, "SVM (IMDb Hyperparameter)")
calculate_mean_median_sd(nb_model_imdb_h, "Naive Bayes (IMDb Hyperparameter)")
calculate_mean_median_sd(rf_model_imdb_h, "Random Forest (IMDb Hyperparameter)")