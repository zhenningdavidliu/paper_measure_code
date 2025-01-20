from models.custom_models import SimpleCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Data loaded.")

    # Initialize model, loss function, and optimizer
    print("Initializing model...")
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Model initialized.")

    batch_size = 64

    # Train the model
    for epoch in tqdm(range(3), desc="Epochs"):  # Number of epochs
        for i, (images, labels) in tqdm(enumerate(train_loader), desc="Batch"):

            optimizer.zero_grad()
            outputs = model(images)

            # Calculate loss    
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = 100 * (predicted == labels).sum().item() / len(labels)

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{10}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    print("Training complete.")

    # Evaluate the model
    print("Evaluating model...")
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Data loaded.")

    losses = []
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Testing"):
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        losses.append(loss.item())
        correct += (predicted == labels).sum().item()
        total += len(labels)

    print(f"Average loss: {sum(losses) / len(losses):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "models/pretrained_mnist_model.pth")
    print("Model trained and saved.")

