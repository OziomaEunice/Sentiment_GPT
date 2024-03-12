import matplotlib.pyplot as plt
import numpy as np

def plot_line_graph(report, title):
    labels = list(report.keys())
    accuracy = [report[label]['accuracy'] for label in labels]
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1_score = [report[label]['f1-score'] for label in labels]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, accuracy, marker='o', label='Accuracy')
    ax.plot(x, precision, marker='o', label='Precision')
    ax.plot(x, recall, marker='o', label='Recall')
    ax.plot(x, f1_score, marker='o', label='F1-Score')

    ax.set_xlabel('Class')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# Example usage:
twitter_report = {
    -1: {'accuracy': 0.6837, 'precision': 0.67, 'recall': 1.00, 'f1-score': 0.80},
    0: {'accuracy': 0.6837, 'precision': 0.78, 'recall': 0.12, 'f1-score': 0.21},
    1: {'accuracy': 0.6837, 'precision': 0.82, 'recall': 0.21, 'f1-score': 0.33}
}

best_model_report = {
    -1: {'accuracy': 0.7579, 'precision': 0.77, 'recall': 0.95, 'f1-score': 0.85},
    0: {'accuracy': 0.7579, 'precision': 0.70, 'recall': 0.35, 'f1-score': 0.47},
    1: {'accuracy': 0.7579, 'precision': 0.71, 'recall': 0.55, 'f1-score': 0.62}
}

plot_line_graph(twitter_report, 'Line Graph for Twitter Dataset')
plot_line_graph(best_model_report, 'Line Graph for Best Model For Twitter Dataset')