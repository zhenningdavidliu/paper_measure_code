import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from foolbox import PyTorchModel, attacks, samples
from foolbox.distances import l2
import numpy as np
from tqdm import tqdm
from models.custom_models import SimpleCNN
import json
import eagerpy as ep
from collections import defaultdict
from constants import mean_l2_distances
import pandas as pd

if __name__ == "__main__":

    # Load config
    with open("config/config_3.json", "r") as f:
        config = json.load(f)
    
    model_name = config["model"]
    model_path = config["model_path"]
    dataset_name = config["dataset"]
    # Step 1: Load MNIST dataset
    if model_name == "vgg16":
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    mean_l2_distance = mean_l2_distances[dataset_name]

    # Step 2: Load a pretrained model for MNIST (here using a simple CNN)


    # Load pretrained weights (if available)
    if model_name == "simple_cnn":
        model = SimpleCNN()
        model.load_state_dict(torch.load("models/pretrained_mnist_model.pth"))
        model.eval()  # Set model to evaluation mode
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
    elif model_name == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
    else:
        raise ValueError(f"Model {model_name} not found")

    # Wrap model with Foolbox for adversarial attack
    fmodel = PyTorchModel(model, bounds=(-1, 1), preprocessing={"mean": 0.5, "std": 0.5})

    # Step 3: Perform adversarial attacks and measure L2 distances
    adversarial_attacks = {
        "FGSM": attacks.FGSM(),
        "DeepFool": attacks.L2DeepFoolAttack(),
        "PGD": attacks.PGD(),
        "L2PGD": attacks.L2PGD(),
    }

    # Make sure epsilons are smaller than mean L2 distance
    epsilons = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    epsilons = [epsilon for epsilon in epsilons if epsilon < mean_l2_distance]

    images, labels = ep.astensors(*samples(fmodel, dataset="mnist", batchsize=16))

    # If the model is colored, we need to repeat the images 3 times
    if model_name in ["resnet18", "vgg16"]:
        images = images.raw.repeat(1, 3, 1, 1)
        if model_name == "vgg16":
            images = transforms.Resize((224, 224))(images)
        images = ep.astensor(images)

    # Only select correctly classified images
    _, predicted = torch.max(fmodel(images).raw.data, 1)
    correct_indices = (labels.raw == predicted).nonzero().flatten()
    images = images[correct_indices]
    labels = labels[correct_indices]
    
    l2_distances = {name: defaultdict(list) for name in adversarial_attacks.keys()}
    min_l2_distances = {name: {batch_idx: mean_l2_distance for batch_idx in range(len(images))} for name in adversarial_attacks.keys()}
    # print(images.shape, labels.shape)
    for name, attack in adversarial_attacks.items():
        print(f"Running {name} attack")
        min_distance = None 
        print(images.shape, labels.shape)
        # try:
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        # print(success)
        # print(len(clipped_advs[0]), images.shape)
        for jdx, adv_examples in enumerate(clipped_advs):
            for idx, (img, adv_example) in enumerate(zip(images, adv_examples)):
                # Compute L2 distance between original and adversarial example
                if success[jdx][idx]:
                    distance = l2(img, adv_example)
                    if len(distance) > 1:
                        distance = np.linalg.norm(distance.raw)
                    if idx not in l2_distances[name]:
                        l2_distances[name][idx] = []
                        l2_distances[name][idx].append(distance.item())
        
        for idx in l2_distances[name]:
            min_l2_distances[name][idx] = min(l2_distances[name][idx])




    # Step 4: Store results as csv
    min_l2_distances_df = pd.DataFrame(min_l2_distances)
    l2_distances_df = pd.DataFrame(l2_distances)
    min_l2_distances_df.to_csv(f"results/min_l2_distances_{model_name}.csv")
    l2_distances_df.to_csv(f"results/l2_distances_{model_name}.csv")
    print(f"Adversarial attacks complete. Results saved as 'l2_distances_{model_name}.npy' and 'min_l2_distances_{model_name}.npy'")
