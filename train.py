from models.custom_models import SimpleCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


if __name__ == "__main__":

    # Check if GPU is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "vgg16"
    if model_name == "simple_cnn":
        greyscale = True
    elif model_name in ["resnet18", "vgg16"]:
        greyscale = False
    else:
        raise ValueError(f"Model name {model_name} not recognized")

    # Load MNIST dataset
    print("Loading data...")
    if model_name == "vgg16":
        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Data loaded.")

    # Initialize model, loss function, and optimizer
    print("Initializing model...")
    if model_name == "simple_cnn":
        model = SimpleCNN()
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_name == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    print("Model initialized.")

    batch_size = 64

    # Train the model
    for epoch in tqdm(range(3), desc="Epochs"):  # Number of epochs
        for i, (images, labels) in tqdm(enumerate(train_loader), desc="Batch"):
            
            if not greyscale:
                images = images.repeat(1, 3, 1, 1)

            images = images.to(device)
            labels = labels.to(device)

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
        if not greyscale:
            images = images.repeat(1, 3, 1, 1)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        losses.append(loss.item())
        correct += (predicted == labels).sum().item()
        total += len(labels)

    print(f"Average loss: {sum(losses) / len(losses):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), f"models/{model_name}_model.pth")
    print("Model trained and saved.")


