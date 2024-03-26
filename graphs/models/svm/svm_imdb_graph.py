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



# SVM results for IMDb
svm_imdb_report = {
    -1: {'precision': 0.56, 'recall': 0.97, 'f1-score': 0.71, 'support': 25000},
    0: {'precision': 0.00, 'recall': 1.00, 'f1-score': 0.00, 'support': 0},
    1: {'precision': 0.91, 'recall': 0.21, 'f1-score': 0.35, 'support': 25000}
}

svm_best_model_imdb_report = {
    -1: {'precision': 0.56, 'recall': 0.97, 'f1-score': 0.71, 'support': 25000},
    0: {'precision': 0.00, 'recall': 1.00, 'f1-score': 0.00, 'support': 0},
    1: {'precision': 0.91, 'recall': 0.21, 'f1-score': 0.35, 'support': 25000}
}

svm_imdb_accuracy = 0.5949
svm_best_model_imdb_accuracy = 0.5949

# Display graphs for IMDb dataset
plot_classification_report(svm_imdb_report, 'SVM Classification Report For IMDb Dataset')
plot_classification_report(svm_best_model_imdb_report, 'SVM Classification Report for Best Model For IMDb Dataset')