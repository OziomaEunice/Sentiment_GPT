import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'SVM vs Naive Bayes': {'accuracy': 4.682682358742056e-10, 'precision': 4.682682358742056e-10, 'recall': 4.682682358742056e-10, 'f1_score': 4.682682358742056e-10},
    'SVM vs Random Forest': {'accuracy': 7.918872654949953e-09, 'precision': 1.552881128220193e-09, 'recall': 2.6596754415408377e-09, 'f1_score': 1.1027236801087996e-09},
    'Naive Bayes vs Random Forest': {'accuracy': 2.083687585314566e-07, 'precision': 2.412288305617491e-08, 'recall': 2.6596754415408377e-09, 'f1_score': 7.427464052378719e-10}
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.plot(kind='bar', rot=0)
plt.ylabel('p-value')
plt.title('Mann-Whitney U Test Results for Twitter Dataset (Hyperparameter)')
plt.yscale('log')  # Log scale for better visualization of small p-values
plt.legend(title='Metric Comparison')
plt.show()