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

# Example usage:
twitter_report = {
    -1: {'precision': 0.67, 'recall': 1.00, 'f1-score': 0.80, 'support': 1835},
    0: {'precision': 0.78, 'recall': 0.12, 'f1-score': 0.21, 'support': 620},
    1: {'precision': 0.82, 'recall': 0.21, 'f1-score': 0.33, 'support': 473}
}

best_model_report = {
    -1: {'precision': 0.77, 'recall': 0.95, 'f1-score': 0.85, 'support': 1835},
    0: {'precision': 0.70, 'recall': 0.35, 'f1-score': 0.47, 'support': 620},
    1: {'precision': 0.71, 'recall': 0.55, 'f1-score': 0.62, 'support': 473}
}

plot_classification_report(twitter_report, 'Classification Report For Twitter Dataset')
plot_classification_report(best_model_report, 'Classification Report for Best Model For Twitter Dataset')