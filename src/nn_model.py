import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        logits = self.conv2d_relu_stack(x)
        return logits


# update one batch
def update_one_batch(model, optimizer, loss_function, images, labels):
    optimizer.zero_grad()
    y_logits = model(images)
    loss = loss_function(y_logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


# model training
def train_one_epoch(model, optimizer, loss_function, train_loader):
    device = next(model.parameters()).device
    model.train()
    losses = []
    total_loss = 0
    samples_seen = 0
    # training one epoch
    for images, labels in iter(train_loader):
        images, labels = images.to(device), labels.to(device)
        batch_loss = update_one_batch(model, optimizer, loss_function, images, labels)
        losses.append(batch_loss)
        actual_batch_size = len(labels)
        total_loss += batch_loss * actual_batch_size
        samples_seen += actual_batch_size
    mean_loss = total_loss / samples_seen
    return (mean_loss, samples_seen, losses)


# model evaluation
def model_eval(model, test_loader, loss_fn=nn.CrossEntropyLoss(reduction="mean")):
    # 1. Switch layer behaviors to evaluation mode
    device = next(model.parameters()).device
    model.eval()
    losses = []
    total_loss = 0
    samples_seen = 0
    # 2. Turn off gradient tracking to save memory
    with torch.no_grad():
        for inputs, labels in iter(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = loss_fn(outputs, labels).item()
            losses.append(batch_loss)
            actual_batch_size = len(labels)
            total_loss += batch_loss * actual_batch_size
            samples_seen += actual_batch_size
    mean_loss = total_loss / samples_seen
    return (mean_loss, samples_seen, losses)
