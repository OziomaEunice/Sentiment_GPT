import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'SVM vs Naive Bayes': {'accuracy': 4.682682358742056e-10, 'precision': 4.682682358742056e-10, 'recall': 4.682682358742056e-10, 'f1_score': 4.682682358742056e-10},
    'SVM vs Random Forest': {'accuracy': 8.006545033944714e-09, 'precision': 3.951377793782832e-09, 'recall': 4.9994348336021215e-09, 'f1_score': 5.680767190281242e-06},
    'Naive Bayes vs Random Forest': {'accuracy': 3.992642741103759e-06, 'precision': 3.951377793782832e-09, 'recall': 6.917385724518674e-08, 'f1_score': 3.729496821025344e-09}
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
df.plot(kind='bar', rot=0)
plt.ylabel('p-value')
plt.title('Mann-Whitney U Test Results for IMDB Dataset (No Hyperparameter)')
plt.yscale('log')  # Log scale for better visualization of small p-values
plt.legend(title='Metric Comparison')
plt.show()