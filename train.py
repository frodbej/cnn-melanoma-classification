import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from melanoma_dataset import MelanomaDataset
from image_transforms import get_transforms
from cnn_model import get_model


def train_model(config):

    print('Training model...')

    # Load training dataset
    train_ds = MelanomaDataset(config['train_data_path'], get_transforms(config['image_size']))

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    # Load model, criterion, optimizer
    model = get_model(config['num_classes'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Iterate over epochs
    for epoch in range(config['epochs']):
        # Set model in training mode
        model.train()

        total_loss = 0

        # Iterate over batches
        for images, labels in train_loader:
            # Reset previous gradients
            optimizer.zero_grad()

            # Forward step, get predictions
            outputs = model(images)

            # Compute loss between predictions and labels
            loss = criterion(outputs, labels)

            # Backpropagration, compute gratient and update model weights
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {round(total_loss/len(train_loader), 4)}")

    # Save model
    torch.save(model.state_dict(), config['model_save_path'])