import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_comparison(accuracies, titles):
    x = np.arange(len(accuracies))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, accuracies, width, label='Accuracy', color='lightgreen')

    ax.set_xlabel('DATASET')
    ax.set_ylabel('ACCURACY')
    ax.set_title('Accuracy Comparison Between Datasets without Fine-tuning')
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

# LLaMA Twitter and IMDb results
nofinetuned_twitter_accuracy = 0.215
nofinetuned_imdb_accuracy = 0.0069

plot_accuracy_comparison([nofinetuned_twitter_accuracy, nofinetuned_imdb_accuracy],
                         ['(No Fine-tuning) Twitter Dataset', '(No Fine-tuning) IMDB Dataset'])