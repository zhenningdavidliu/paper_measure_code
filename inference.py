import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from foolbox import PyTorchModel, attacks
from foolbox.distances import l2
import numpy as np
from tqdm import tqdm

# Step 1: Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Step 2: Load a pretrained model for MNIST (here using a simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load pretrained weights (if available)
model = SimpleCNN()
model.load_state_dict(torch.load("pretrained_mnist_model.pth"))
model.eval()  # Set model to evaluation mode

# Wrap model with Foolbox for adversarial attack
fmodel = PyTorchModel(model, bounds=(-1, 1), preprocessing=(0.5, 0.5))

# Step 3: Perform adversarial attacks and measure L2 distances
adversarial_attacks = {
    "FGSM": attacks.FGSM(),
    "DeepFool": attacks.L2DeepFoolAttack(),
    "CarliniWagner": attacks.L2CarliniWagnerAttack(),
}

l2_distances = {name: [] for name in adversarial_attacks.keys()}

for images, labels in tqdm(test_loader, desc="Evaluating on MNIST"):
    images, labels = images.numpy(), labels.numpy()
    
    for name, attack in adversarial_attacks.items():
        try:
            adv_example = attack(fmodel, images[0], labels[0])
            if adv_example is not None:
                # Compute L2 distance between original and adversarial example
                distance = l2(images[0], adv_example)
                l2_distances[name].append(distance)
            else:
                l2_distances[name].append(None)  # Attack failed
        except Exception as e:
            print(f"Attack {name} failed for one example: {e}")
            l2_distances[na
