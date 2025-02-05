import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # (28x28 → 28x28)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # (28x28 → 28x28)
        
        # Pooling layer (2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # (28x28 → 14x14)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0-9)
    
    def forward(self, x):     # x.size = Batch_size * 1*28*28
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 → ReLU → Pool   x.size = Batch_size * 32*14*14
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 → ReLU → Pool    x.size = Batch_size * 64*7*7
        x = x.view(-1, 64 * 7 * 7)  # Flatten       x.size = Batch_size * 3136
        x = F.relu(self.fc1(x))  # Fully connected layer 1      x.size = Batch_size * 128
        x = self.fc2(x)  # Fully connected layer 2 (output)     x.size = Batch_size * 10
        return x

# Instantiate model
model = CNN()
print(model)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1]
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        # print(labels.size())
        # print(outputs.size())
        loss = criterion(outputs, labels)  # Compute loss     |  label.size = Batch_size,   outputs.size = Batch_size * 10
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

correct = 0
total = 0
with torch.no_grad():  # No gradient calculation needed
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

