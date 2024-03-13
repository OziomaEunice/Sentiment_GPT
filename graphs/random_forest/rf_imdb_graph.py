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



# Classification Reports and Accuracy for RF
rf_report = {
    -1: {'precision': 0.60, 'recall': 0.75, 'f1-score': 0.67, 'support': 25000},
    0: {'precision': 0.00, 'recall': 1.00, 'f1-score': 0.00, 'support': 0},
    1: {'precision': 0.82, 'recall': 0.30, 'f1-score': 0.44, 'support': 25000}
}

best_model_rf_report = {
    -1: {'precision': 0.58, 'recall': 0.80, 'f1-score': 0.67, 'support': 25000},
    0: {'precision': 0.00, 'recall': 1.00, 'f1-score': 0.00, 'support': 0},
    1: {'precision': 0.83, 'recall': 0.28, 'f1-score': 0.42, 'support': 25000}
}


# Display graphs for RF results
plot_classification_report(rf_report, 'Classification Report For RF Model on IMDb Dataset')
plot_classification_report(best_model_rf_report, 'Classification Report for Best RF Model on IMDb Dataset')