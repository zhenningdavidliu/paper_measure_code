import torch
import torchvision
import torchvision.transforms as transforms
from foolbox.distances import l2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    dataset_name = "mnist"

    # Load whole MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    images_train = train_dataset.data.view(train_dataset.data.size(0), -1)
    images_test = test_dataset.data.view(test_dataset.data.size(0), -1)

    images = torch.cat((images_train, images_test), 0)

    # Calculate mean L2 distance for whole dataset between closest images
    min_l2_distances = 0
    for i in tqdm(range(len(images)), desc="Calculating mean L2 distance"):
        # Remove current image from the list
        image = images[i]
        temp_images = images.clone()
        temp_images = torch.cat((temp_images[:i], temp_images[i+1:]), 0)

        min_dist = np.min(l2(image, temp_images).numpy())
        min_l2_distances += min_dist
    min_l2_distances /= len(images)
    print(f"Mean L2 distance: {min_l2_distances}")
    np.save(f"constants", min_l2_distances)