# src/multi_classification.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

import torch.nn as nn



# Import pytorch and other libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

# Define the number of classes
num_classes = 5 

# Define the CNN model
class ECG_CNN(nn.Module):
    def __init__(self):
        super(ECG_CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        # self.fc1 = nn.Linear(in_features=64*125, out_features=128) 
        self.fc1 = nn.Linear(in_features=16384, out_features=128)

        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Pass the output through the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Apply softmax to get the probabilities
        x = self.softmax(x)
        return x


def train_classifier(classifier, train_loader, val_loader, num_epochs, device, patience=5):
    
    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(classifier.parameters(), lr=0.01) 

    # Define the number of epochs
    training_losses = []
    validation_accuracies = []
    # Loop over the epochs
    for epoch in range(num_epochs):


        # Initialize the running loss value
        running_loss = 0.0

        # Loop over the batches of data from the training loader
        for i, data in enumerate(train_loader, 0):

            # Get the input data and the true labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # print("Input data shape:", inputs.shape)
            # print("labels data shape:", labels.shape)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Feed the input data to the model and get the output predictions
            outputs = classifier(inputs)

            # Compare the predictions with the true labels and compute the loss value
            loss = criterion(outputs, labels)

            # Backpropagate the loss value and update the model parameters using the optimizer
            loss.backward()
            optimizer.step()

            # Add the loss value to the running loss value
            running_loss += loss.item()

            # Print the average loss value every 2000 batches
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # After each epoch, evaluate the model performance on the validation data
        # Initialize the correct and total counts
        correct = 0
        total = 0

        # Loop over the batches of data from the validation loader
        with torch.no_grad(): # No need to compute gradients for validation
            for data in val_loader:
                # Get the input data and the true labels
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # Feed the input data to the model and get the output predictions
                outputs = classifier(inputs)
                # Get the predicted labels by taking the argmax of the outputs
                _, predicted = torch.max(outputs.data, 1)
                # Update the correct and total counts
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        validation_accuracy = 100 * correct / total
        validation_accuracies.append(validation_accuracy)
        # Print the accuracy value for the current epoch
        print('Accuracy of the model on the validation data: %d %%' % (100 * correct / total))

    return training_losses, validation_accuracies


# Function to evaluate the classifier with additional metrics
def evaluate_classifier(classifier, test_loader, device):
    classifier.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Classification Report (Precision, Recall, F1-score)
    cr = classification_report(y_true, y_pred)
    print("Classification Report:\n", cr)

    return accuracy
