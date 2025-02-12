
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to fit ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])

# Load a sample dataset (replace with your own dataset)
train_data = datasets.FakeData(transform=transform)  # Example dataset
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Load Pretrained ResNet18 Model
model = models.resnet18(pretrained=True)
print(model.fc)
# Modify the last fully connected layer for a new classification task (e.g., 10 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Change output classes
print(model.fc)
# print(model.parameters())
# print(model.fc.parameters())
# Freeze all layers except the classifier (Optional)
for param in model.parameters():
    print(param)
    param.requires_grad = False  # Freeze pretrained layers

for param in model.fc.parameters():
    print(param)
    param.requires_grad = True  # Train only the last layer

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "transfer_learning_model.pth")
print("Training complete & model saved.")
