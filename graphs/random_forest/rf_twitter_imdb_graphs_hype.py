import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_comparison(accuracies, titles):
    x = np.arange(len(accuracies))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, accuracies, width, label='Accuracy', color='lightblue')

    ax.set_xlabel('DATASET')
    ax.set_ylabel('ACCURACY')
    ax.set_title('Accuracy Comparison Between Datasets with Hyperparameter')
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

# RF Twitter and IMDb results
best_model_twitter_accuracy = 0.7643
best_model_imdb_accuracy = 0.4981

plot_accuracy_comparison([best_model_twitter_accuracy, best_model_imdb_accuracy],
                         ['Best Model (Twitter)', 'Best Model (IMDb)'])