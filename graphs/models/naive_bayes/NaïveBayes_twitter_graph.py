import matplotlib.pyplot as plt
import numpy as np

def plot_classification_report(report, title):
    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

    precision = [report[label]['precision'] for label in report.keys()]
    recall = [report[label]['recall'] for label in report.keys()]
    f1_score = [report[label]['f1-score'] for label in report.keys()]
    support = [report[label]['support'] for label in report.keys()]

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

# Naive Bayes results for Twitter US Airline  
twitter_report = {
    -1: {'precision': 0.67, 'recall': 1.00, 'f1-score': 0.80, 'support': 4589},
    0: {'precision': 0.79, 'recall': 0.10, 'f1-score': 0.18, 'support': 1550},
    1: {'precision': 0.89, 'recall': 0.18, 'f1-score': 0.30, 'support': 1181}
}

best_model_report = {
    -1: {'precision': 0.76, 'recall': 0.95, 'f1-score': 0.85, 'support': 4589},
    0: {'precision': 0.70, 'recall': 0.32, 'f1-score': 0.44, 'support': 1550},
    1: {'precision': 0.71, 'recall': 0.52, 'f1-score': 0.60, 'support': 1181}
}

plot_classification_report(twitter_report, 'Classification Report For Twitter Dataset')
plot_classification_report(best_model_report, 'Classification Report for Best Model For Twitter Dataset')