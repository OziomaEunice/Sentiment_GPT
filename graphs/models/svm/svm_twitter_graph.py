import matplotlib.pyplot as plt
import numpy as np

def plot_classification_report(report, title):
    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

    precision = [report[label]['precision'] for label in report.keys()]
    recall = [report[label]['recall'] for label in report.keys()]
    f1_score = [report[label]['f1-score'] for label in report.keys()]

    labels = [sentiment_labels[label] for label in report.keys()]

    x = np.arange(len(labels))
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1-Score')

    ax.set_xlabel('SENTIMENT')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()



# SVM results for Twitter
svm_twitter_report = {
    -1: {'precision': 0.81, 'recall': 0.92, 'f1-score': 0.86, 'support': 1835},
    0: {'precision': 0.66, 'recall': 0.50, 'f1-score': 0.57, 'support': 620},
    1: {'precision': 0.78, 'recall': 0.63, 'f1-score': 0.69, 'support': 473}
}

svm_best_model_twitter_report = {
    -1: {'precision': 0.81, 'recall': 0.92, 'f1-score': 0.86, 'support': 1835},
    0: {'precision': 0.66, 'recall': 0.50, 'f1-score': 0.57, 'support': 620},
    1: {'precision': 0.78, 'recall': 0.63, 'f1-score': 0.69, 'support': 473}
}

svm_twitter_accuracy = 0.7845
svm_best_model_twitter_accuracy = 0.7845

# Display graphs for Twitter dataset
plot_classification_report(svm_twitter_report, 'SVM Classification Report For Twitter Dataset')
plot_classification_report(svm_best_model_twitter_report, 'SVM Classification Report for Best Model For Twitter Dataset')