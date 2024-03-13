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

'''
def plot_accuracy_comparison(accuracies, titles):
    x = np.arange(len(accuracies))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, accuracies, width, label='Accuracy', color='lightblue')

    ax.set_xlabel('DATASET')
    ax.set_ylabel('ACCURACY')
    ax.set_title('Accuracy Comparison Between Datasets (IMDB)')
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.legend()

    # Add data labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate('%.2f%%' % (height * 100),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()
'''

# Naive Bayes results for IMDb 
imdb_report = {
    -1: {'precision': 0.50, 'recall': 1.00, 'f1-score': 0.67, 'support': 25000},
    0: {'precision': 0.00, 'recall': 1.00, 'f1-score': 0.00, 'support': 0},
    1: {'precision': 0.95, 'recall': 0.02, 'f1-score': 0.03, 'support': 25000}
}

best_model_imdb_report = {
    -1: {'precision': 0.57, 'recall': 0.94, 'f1-score': 0.71, 'support': 25000},
    0: {'precision': 0.00, 'recall': 1.00, 'f1-score': 0.00, 'support': 0},
    1: {'precision': 0.85, 'recall': 0.23, 'f1-score': 0.36, 'support': 25000}
}

imdb_accuracy = 0.5077
best_model_imdb_accuracy = 0.5845

# Display graphs for IMDb dataset
plot_classification_report(imdb_report, 'Classification Report For IMDb Dataset')
plot_classification_report(best_model_imdb_report, 'Classification Report for Best Model For IMDb Dataset')
# plot_accuracy_comparison([imdb_accuracy, best_model_imdb_accuracy],
#                          ['IMDb Dataset', 'Best Model IMDb Dataset'])