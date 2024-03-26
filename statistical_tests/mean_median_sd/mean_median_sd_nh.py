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




#------------Twitter US Airline Sentiment Dataset (No Hyperparameter)------------#
# svm model
svm_model_twitter_nh = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
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












# Perform calculations for each model on the datasets
calculate_mean_median_sd(svm_model_twitter_nh, "SVM (Twitter No Hyperparameter)")
calculate_mean_median_sd(nb_model_twitter_nh, "Naive Bayes (Twitter No Hyperparameter)")
calculate_mean_median_sd(rf_model_twitter_nh, "Random Forest (Twitter No Hyperparameter)")
# calculate_mean_median_sd(svm_model_imdb_nh, "SVM (IMDb No Hyperparameter)")
# calculate_mean_median_sd(nb_model_imdb_nh, "Naive Bayes (IMDb No Hyperparameter)")
# calculate_mean_median_sd(rf_model_imdb_nh, "Random Forest (IMDb No Hyperparameter)")