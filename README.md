# Simple CNN for Melanoma Image Classification 🧠🔬

This project implements a simple Convolutional Neural Network (CNN) for binary classification of melanoma skin cancer images using PyTorch.

## Dataset

The dataset used comes from Kaggle:  
**[Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)**

To automatically download and prepare the dataset, run the following script:

```bash
bash download_data.sh
```

## Configuration

The model is defined in `cnn_model.py`.

All key settings are defined in `config.yaml`:

```yaml
num_classes: 2
image_size: 64
batch_size: 32
epochs: 10
learning_rate: 0.001
train_data_path: 'melanoma_cancer_dataset/train'
test_data_path: 'melanoma_cancer_dataset/test'
model_save_path: 'melanoma_model.pth'
mode: 'train'  # train or test
```

To train the model, run:

```bash
python main.py
```

To test the trained model, switch to test mode in the `config.yaml` and run again:

```bash
python main.py
```
