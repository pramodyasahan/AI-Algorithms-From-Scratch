import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from cnn import SimpleCNN
import argparse


# Cross Entropy Loss
def cross_entropy_loss(output, target):
    # Batch size (number of examples)
    m = output.shape[0]

    # Compute softmax probabilities
    p = np.exp(output - np.max(output, axis=1, keepdims=True))
    probs = p / np.sum(p, axis=1, keepdims=True)

    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    probs = np.clip(probs, epsilon, 1.0 - epsilon)  # Clipping to avoid probabilities being exactly 0 or 1

    # Flatten target and ensure it’s an integer
    target = target.astype(int).flatten()

    # Ensure proper shape for both probs and target
    assert probs.shape[0] == target.shape[0], "Shape mismatch between probs and target"

    loss = -np.mean(np.log(probs[np.arange(m), target]))
    return loss


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def train(model, train_loader, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        print(f"Training on epoch {epoch + 1}/{epochs}")

        # Using tqdm for progress bar during training
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (inputs, targets) in progress_bar:
            inputs = tensor_to_numpy(inputs)
            targets = tensor_to_numpy(targets).astype(int)

            outputs = model.forward(inputs)

            loss = cross_entropy_loss(outputs, targets)

            d_output = np.zeros_like(outputs)
            d_output[range(len(targets)), targets] = -1 / len(targets)
            model.backward(d_output, learning_rate)

            total_loss += loss

            preds = np.argmax(outputs, axis=1)
            correct += np.sum(preds == targets)
            total += targets.shape[0]

            # Update tqdm progress bar with current loss and accuracy
            progress_bar.set_postfix(loss=loss, accuracy=correct / total)


def test(model, test_loader):
    correct = 0
    total = 0

    # Progress bar for testing
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")

    for batch_idx, (inputs, targets) in progress_bar:
        inputs = tensor_to_numpy(inputs)
        targets = tensor_to_numpy(targets).astype(int)

        outputs = model.forward(inputs)
        preds = np.argmax(outputs, axis=1)
        correct += np.sum(preds == targets)
        total += targets.shape[0]

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a Simple CNN on CIFAR-10 Dataset")

    # Adding command-line arguments
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer",
                        required=True)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training", required=True)
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model", required=True)
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory to download/load CIFAR-10 dataset",
                        required=True)

    # Parse arguments from command line
    args = parser.parse_args()

    # Hyperparameters from command-line arguments
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    data_dir = args.data_dir

    # CIFAR-10 Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SimpleCNN()

    # Train the model
    train(model, train_loader, learning_rate, epochs)

    # Test the model
    test(model, test_loader)
