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

# Llama results for Twitter dataset
llama_twitter_report_no_finetuning = {
    -1: {'precision': 0.80, 'recall': 0.00, 'f1-score': 0.00, 'support': 4589},
    0: {'precision': 0.20, 'recall': 0.89, 'f1-score': 0.33, 'support': 1550},
    1: {'precision': 0.39, 'recall': 0.16, 'f1-score': 0.23, 'support': 1182}
}

llama_twitter_report_finetuned = {
    -1: {'precision': 0.89, 'recall': 0.92, 'f1-score': 0.90, 'support': 4589},
    0: {'precision': 0.69, 'recall': 0.65, 'f1-score': 0.67, 'support': 1550},
    1: {'precision': 0.81, 'recall': 0.75, 'f1-score': 0.78, 'support': 1182}
}

# Display graphs for Llama results on Twitter dataset
plot_classification_report(llama_twitter_report_no_finetuning, 'Classification Report For Twitter Dataset (No Fine-tuning)')
plot_classification_report(llama_twitter_report_finetuned, 'Classification Report For Twitter Dataset (Fine-tuned)')