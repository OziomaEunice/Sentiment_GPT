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

# Llama results for IMDb dataset
llama_imdb_report_no_finetuning = {
    -1: {'precision': 1.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 12500},
    0: {'precision': 0.00, 'recall': 0.00, 'f1-score': 0.00, 'support': 0},
    1: {'precision': 0.88, 'recall': 0.01, 'f1-score': 0.03, 'support': 12500}
}

llama_imdb_report_finetuned = {
    -1: {'precision': 0.96, 'recall': 0.96, 'f1-score': 0.96, 'support': 12500},
    1: {'precision': 0.96, 'recall': 0.96, 'f1-score': 0.96, 'support': 12500}
}

# Display graphs for Llama results
plot_classification_report(llama_imdb_report_no_finetuning, 'Classification Report For IMDb Dataset (No Fine-tuning)')
plot_classification_report(llama_imdb_report_finetuned, 'Classification Report For IMDb Dataset (Fine-tuned)')