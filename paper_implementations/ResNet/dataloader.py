from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def dataloader_cifar():
    # Correct normalization for CIFAR-10 (RGB images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10('/content/drive/MyDrive/All_Datasets/CIFAR10', train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10('/content/drive/MyDrive/All_Datasets/CIFAR10', train=False, download=True,
                                    transform=transform)

    # Split dataset into training and validation sets
    train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])

    # Print dataset information
    print(f"Image shape of a random sample image: {train_dataset[0][0].numpy().shape}\n")
    print(f"Training Set:   {len(train_dataset)} images")
    print(f"Validation Set: {len(val_dataset)} images")
    print(f"Test Set:       {len(test_dataset)} images")

    BATCH_SIZE = 32

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)  # Usually no need to shuffle the test set

    return train_loader, val_loader, test_loader
