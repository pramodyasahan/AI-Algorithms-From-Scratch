import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from dataloader import dataloader_cifar
from layers import BasicConvBlock
from paper_implementations.ResNet.resnet import ResNet


# Function to create a ResNet-56 model
def ResNet56():
    return ResNet(block_type=BasicConvBlock, num_blocks=[9, 9, 9])


# Function to train the model
def train_model(model, device, train_loader, val_loader, criterion, optimizer, epochs=15):
    train_samples_num = len(train_loader.dataset)
    val_samples_num = len(val_loader.dataset)
    train_costs, val_costs = [], []

    # Training phase
    for epoch in range(epochs):
        train_running_loss = 0
        correct_train = 0

        model.train()  # Set the model to training mode

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            prediction = model(inputs)  # Forward pass

            loss = criterion(prediction, labels)  # Compute the loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            _, predicted_outputs = torch.max(prediction.data, 1)
            correct_train += (predicted_outputs == labels).float().sum().item()
            train_running_loss += loss.item() * inputs.size(0)

        train_epoch_loss = train_running_loss / train_samples_num
        train_costs.append(train_epoch_loss)
        train_acc = correct_train / train_samples_num

        # Validation phase
        val_running_loss = 0
        correct_val = 0
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                prediction = model(inputs)  # Forward pass
                loss = criterion(prediction, labels)  # Compute the loss

                _, predicted_outputs = torch.max(prediction.data, 1)
                correct_val += (predicted_outputs == labels).float().sum().item()

                val_running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / val_samples_num
        val_costs.append(val_epoch_loss)
        val_acc = correct_val / val_samples_num

        # Logging the results
        info = "[Epoch {}/{}]: train-loss = {:.6f} | train-acc = {:.3f} | val-loss = {:.6f} | val-acc = {:.3f}"
        print(info.format(epoch + 1, epochs, train_epoch_loss, train_acc, val_epoch_loss, val_acc))

        # Save the model checkpoint
        torch.save(model.state_dict(), f'./checkpoint_gpu_{epoch + 1}.pth')

    # Save final model weights
    torch.save(model.state_dict(), './resnet-56_weights_gpu.pth')

    return train_costs, val_costs


# Main function
def main():
    # Initialize model, device, criterion, and optimizer
    model = ResNet56()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (3, 32, 32))

    # Load data
    train_loader, val_loader, test_loader = dataloader_cifar()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train_costs, val_costs = train_model(model, device, train_loader, val_loader, criterion, optimizer)

    # Load the best model weights
    model.load_state_dict(torch.load('./resnet-56_weights_gpu.pth'))


# Entry point
if __name__ == "__main__":
    main()
