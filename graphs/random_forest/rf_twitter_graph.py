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



# RF results for Twitter
twitter_rf_report = {
    -1: {'precision': 0.79, 'recall': 0.92, 'f1-score': 0.85, 'support': 1835},
    0: {'precision': 0.63, 'recall': 0.47, 'f1-score': 0.54, 'support': 620},
    1: {'precision': 0.76, 'recall': 0.55, 'f1-score': 0.64, 'support': 473}
}

best_model_twitter_rf_report = {
    -1: {'precision': 0.79, 'recall': 0.92, 'f1-score': 0.85, 'support': 1835},
    0: {'precision': 0.64, 'recall': 0.47, 'f1-score': 0.54, 'support': 620},
    1: {'precision': 0.76, 'recall': 0.55, 'f1-score': 0.64, 'support': 473}
}

# Display graphs for Twitter RF results
plot_classification_report(twitter_rf_report, 'Classification Report For RF Model on Twitter Dataset')
plot_classification_report(best_model_twitter_rf_report, 'Classification Report for Best RF Model on Twitter Dataset')