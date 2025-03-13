# Import necessary libraries
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Step 1: Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(            # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Step 2: Load the dataset
dataset_path = "/home/sundar/Rice-Classification/Rice_Image_Dataset"  # Update this path
print("Loading dataset from:", dataset_path)
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
print("Class names:", dataset.classes)
print("Number of images:", len(dataset))

# Step 3: Split the dataset into training and validation sets
subset_size = int(0.1 * len(dataset))  # Use only 10% of the dataset
print("Using subset size:", subset_size)
subset_dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

train_size = int(0.8 * len(subset_dataset))
val_size = len(subset_dataset) - train_size
print("Splitting dataset into training and validation sets...")
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])
print("Training set size:", len(train_dataset))
print("Validation set size:", len(val_dataset))

# Step 4: Create data loaders
print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("Data loaders created successfully!")

# Step 5: Define the model
print("Defining model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)
print("Model defined successfully!")

# Step 6: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train the model
num_epochs = 2  # Train for only 2 epochs
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print progress every 10 batches
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {running_loss/i:.4f}")
    
    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")
print("Training completed!")

# Step 8: Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Step 9: Save the model
torch.save(model.state_dict(), "rice_classifier.pth")
print("Model saved as 'rice_classifier.pth'.")