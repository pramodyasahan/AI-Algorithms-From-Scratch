import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from dataloader import dataloader_cifar
from layers import BasicConvBlock
from ..ResNet.resnet import ResNet


def ResNet56():
    return ResNet(block_type=BasicConvBlock, num_blocks=[9, 9, 9])


model = ResNet56()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = 'cpu'
model.to(device)
summary(model, (3, 32, 32))

train_loader, val_loader, test_loader = dataloader_cifar()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train_model():
    EPOCHS = 15
    train_samples_num = 45000
    val_samples_num = 5000
    train_costs, val_costs = [], []

    # Training phase.
    for epoch in range(EPOCHS):

        train_running_loss = 0
        correct_train = 0

        model.train().cuda()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            prediction = model(inputs)

            loss = criterion(prediction, labels)

            loss.backward()
            optimizer.step()

            _, predicted_outputs = torch.max(prediction.data, 1)

            # Update the running corrects
            correct_train += (predicted_outputs == labels).float().sum().item()

            train_running_loss += (loss.data.item() * inputs.shape[0])

        train_epoch_loss = train_running_loss / train_samples_num

        train_costs.append(train_epoch_loss)

        train_acc = correct_train / train_samples_num

        # Now check trained weights on the validation set
        val_running_loss = 0
        correct_val = 0

        model.eval().cuda()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass.
                prediction = model(inputs)

                # Compute the loss.
                loss = criterion(prediction, labels)

                # Compute validation accuracy.
                _, predicted_outputs = torch.max(prediction.data, 1)
                correct_val += (predicted_outputs == labels).float().sum().item()

            # Compute batch loss.
            val_running_loss += (loss.data.item() * inputs.shape[0])

            val_epoch_loss = val_running_loss / val_samples_num
            val_costs.append(val_epoch_loss)
            val_acc = correct_val / val_samples_num

        info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f} | val-loss = {:0.6f} | val-acc = {:0.3f}"

        print(info.format(epoch + 1, EPOCHS, train_epoch_loss, train_acc, val_epoch_loss, val_acc))

        torch.save(model.state_dict(), '/content/checkpoint_gpu_{}'.format(epoch + 1))

    torch.save(model.state_dict(), '/content/resnet-56_weights_gpu')

    return train_costs, val_costs


train_costs, val_costs = train_model()
model = ResNet56()
model.load_state_dict(torch.load('/content/resnet-56_weights_gpu'))

test_samples_num = 256
correct = 0

model.eval().cuda()

with  torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Make predictions.
        prediction = model(inputs)

        # Retrieve predictions indexes.
        _, predicted_class = torch.max(prediction.data, 1)

        # Compute number of correct predictions.
        correct += (predicted_class == labels).float().sum().item()

test_accuracy = correct / test_samples_num
print('Test accuracy: {}'.format(test_accuracy))
