import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

from melanoma_dataset import MelanomaDataset
from image_transforms import get_transforms
from cnn_model import get_model


def test_model(config):

    # Raise error if there is no saved model
    if not os.path.exists(config['model_save_path']):
        raise FileNotFoundError('The model is not found.')

    print('Testing model...')

    # Load trained model
    model = get_model(config['num_classes'])
    model.load_state_dict(torch.load(config['model_save_path']))

    # Set eval mode
    model.eval()

    # Load test data
    test_ds = MelanomaDataset(config['test_data_path'], get_transforms(config['image_size']))

    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad(): # Disable gradient calculation for testing
        # Iterate over batches
        for images, labels in test_loader:

            # Get predictions
            outputs = model(images)

            # Get class with higher probability
            preds = torch.argmax(outputs, dim=1)

            # Convert tensor to numpy and extend
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    # Compute accuracy of the model
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {round(acc*100, 2)}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')
    plt.show()