import torch
import torch.nn as nn
import torchvision
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


if __name__ == "__main__":

    # Load config
    with open("config/config_1.json", "r") as f:
        config = json.load(f)
    
    model_name = config["model"]
    model_path = config["model_path"]
    dataset_name = config["dataset"]
    # Step 1: Load MNIST dataset
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
    
    l2_distances = {name: defaultdict(list) for name in adversarial_attacks.keys()}
    min_l2_distances = {name: {batch_idx: mean_l2_distance for batch_idx in range(len(images))} for name in adversarial_attacks.keys()}
    # print(images.shape, labels.shape)
    for name, attack in adversarial_attacks.items():
        min_distance = None 
        # try:
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        # print(success)
        # print(len(clipped_advs[0]), images.shape)
        for jdx, adv_examples in enumerate(clipped_advs):
            for idx, (img, adv_example) in enumerate(zip(images, adv_examples)):
                # Compute L2 distance between original and adversarial example
                if success[jdx][idx]:
                    distance = l2(img, adv_example)
                    if idx not in l2_distances[name]:
                        l2_distances[name][idx] = []
                        l2_distances[name][idx].append(distance.item())
        
        for idx in l2_distances[name]:
            min_l2_distances[name][idx] = min(l2_distances[name][idx])




    # Step 4: Store results
    print(min_l2_distances)
    np.save(f"results/l2_distances_{model_name}.npy", l2_distances)
    np.save(f"results/min_l2_distances_{model_name}.npy", min_l2_distances)
    print(f"Adversarial attacks complete. Results saved as 'l2_distances_{model_name}.npy' and 'min_l2_distances_{model_name}.npy'")
