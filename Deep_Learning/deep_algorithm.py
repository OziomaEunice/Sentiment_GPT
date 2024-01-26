# Feedforward Neural Network for Sentiment Analysis


# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (replace with your dataset)
data = [("I love this product", "positive"),
        ("Not bad, but could be better", "neutral"),
        ("This is a terrible experience", "negative")]

# Extract features (Bag of Words representation)
texts, labels = zip(*data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Convert labels to numerical values
label_dict = {"negative": 0, "neutral": 1, "positive": 2}
y = [label_dict[label] for label in labels]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Define a simple Feedforward Neural Network
class SentimentModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SentimentModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# Initialize the model, loss function, and optimizer
model = SentimentModel(input_size=X_train.shape[1], output_size=len(label_dict))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions on the test set
with torch.no_grad():
    model.eval()
    predictions = torch.argmax(model(X_test_tensor), dim=1).numpy()

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, predictions))
