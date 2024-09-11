import torch
import torch.nn as nn
import torch.optim as optim

# Example dataset: a simple string
data = "They come in waves, my feelings for you. And not pretty whitecaps dancing at my feet."

# Creating character mappings
chars = list(set(data))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert the data into integers
data_idx = [char_to_idx[ch] for ch in data]
vocab_size = len(chars)

# Hyperparameters
input_size = vocab_size
hidden_size = 16
output_size = vocab_size
learning_rate = 0.01
num_epochs = 1000
seq_length = 20


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # Define the layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Reshape input to match batch size of 1
        input = input.view(1, -1)
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# Instantiate the RNN model
rnn = RNN(input_size, hidden_size, output_size)

# Define loss function (Negative Log-Likelihood Loss)
criterion = nn.NLLLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    hidden = rnn.init_hidden()
    loss = 0

    # Prepare the input and target sequences
    for i in range(len(data_idx) - seq_length):
        input_seq = data_idx[i:i + seq_length]
        target_seq = data_idx[i + 1:i + seq_length + 1]

        # Convert to one-hot encoded tensors
        input_tensor = torch.zeros(seq_length, input_size)
        target_tensor = torch.tensor(target_seq)

        for t in range(seq_length):
            input_tensor[t][input_seq[t]] = 1

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        hidden = rnn.init_hidden()  # Reset hidden state at each new sequence

        # Loop through the sequence
        for t in range(seq_length):
            output, hidden = rnn(input_tensor[t], hidden)  # Pass one character at a time

        loss = criterion(output, target_tensor[-1].view(1))
        loss.backward()
        optimizer.step()

    # Print the loss for each epoch
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")


def predict(input_char, hidden=None):
    input_tensor = torch.zeros(1, input_size)
    input_tensor[0][char_to_idx[input_char]] = 1

    if hidden is None:
        hidden = rnn.init_hidden()

    with torch.no_grad():
        output, hidden = rnn(input_tensor, hidden)

    output_char = idx_to_char[torch.argmax(output).item()]
    return output_char, hidden


start_char = 'T'
hidden = None
predicted = start_char

# Predict the next characters
for _ in range(10):
    next_char, hidden = predict(predicted[-1], hidden)
    predicted += next_char

print("Predicted string:", predicted)
