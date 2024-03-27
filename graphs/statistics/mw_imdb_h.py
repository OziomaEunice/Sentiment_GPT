import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'SVM vs Naive Bayes': {'accuracy': 4.682682358742056e-10, 'precision': 1.0, 'recall': 4.682682358742056e-10, 'f1_score': 4.682682358742056e-10},
    'SVM vs Random Forest': {'accuracy': 8.006545033944714e-09, 'precision': 4.193185625222832e-09, 'recall': 4.536019325117111e-09, 'f1_score': 6.349117418167286e-06},
    'Naive Bayes vs Random Forest': {'accuracy': 8.006545033944714e-09, 'precision': 4.193185625222832e-09, 'recall': 4.536019325117111e-09, 'f1_score': 4.344501820937825e-09}
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.plot(kind='bar', rot=0)
plt.ylabel('p-value')
plt.title('Mann-Whitney U Test Results for IMDB Dataset (Hyperparameter)')
plt.yscale('log')  # Log scale for better visualization of small p-values
plt.legend(title='Metric Comparison')
plt.show()