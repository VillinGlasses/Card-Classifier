# First-ever Neural Network deep learning model
# Takes input of the images of playing cards and identifies the card

# Imports PyTorch machine learning framework
import torch

# FUNCTIONALITY IMPORTS FROM PYTORCH
import torch.nn as nn                               # torch.nn module from PyTorch contains helpers for creating a neural network
import torch.optim as optim                         # torch.optim module is a package that contains optimization algorithms to correct the neural network. This helps it "learn".
from torch.utils.data import Dataset, DataLoader    # Dataset and DataLoader are both important for getting data for the machine learning model.
import torchvision.transforms as transforms         # torchvision from PyTorch contains a package that allows the neural network to process visual data.
from torchvision.datasets import ImageFolder        # transforms is a module that provides a set of common image transformations for preprocessing and augmenting image data. ImageFolder is a class derived from torchvision.datasets that helps us load image data from a directory structure.
import timm                                         # TIMM is a library for image classification within PyTorch

# DATA VISUALIZATION TOOLS
import matplotlib.pyplot as plt                     # matplotlib is a comprehensive library for data visualization. Pyplot is a collection of functions that provide an interface for it.
import pandas as pd                                 # pandas is an open-source data analysis and manipulation library for Python.
import numpy as np                                  # numpy is a python library for working with arrays.
import sys                                          # sys provides functions and variables which are used to manipulate the Python Runtime Environment.
from tqdm import tqdm                               # wrapper that will allow us to make a progress bar for each phase of training and validation
from PIL import Image                               # Imports Pillow, allowing us to use Python Imaging for test outputs
from glob import glob                               # Imports glob, which allows us to grab all file paths that get at least a partial match with what is specified

# Class PlayingCardDataset inherits from the Dataset from the PyTorch utils file.
# Organizes the data so that we can load it into the neural network model.
class PlayingCardDataset(Dataset):
    # init takes in the name of the data directory, and has a default transform value of None.
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform)  # ImageFolder class object initialized with data_dir and transform parameters.

    # Tells the data loader how many examples we have in the dataset.
    def __len__(self):
        return len(self.data)
    
    # Item retrieval function for our dataset. Takes index location idx as an input, returns the data item at that index.
    def __getitem__(self, idx):
        return self.data[idx]
    
    # @property decorator defines methods that act like attributes.
    # classes method returns the data classes from the ImageFolder object called self.data
    @property
    def classes(self):
        return self.data.classes


# CardClassifier class inherits from the nn Module from PyTorch.
# Uses built-in architecture from the TIMM image classification library to identify the visual inputs.
class CardClassifier(nn.Module):
    # defining all parts of the model. We have 53 classes because we account for the Joker card.
    def __init__(self, num_classes = 53):
        super(CardClassifier, self).__init__()                                      # Super Call to the Parent nn.Module class, allows us to initialize with members of the parent class.
        self.base_model = timm.create_model('efficientnet_b0', pretrained = True)   # uses the efficientnet_b0 model because our dataset is small. It has already been pretrained on the ImageNet data
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Making a classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    # Defines how the model is connected. Takes in a data batch and returns classifier as an output variable
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# Variables to hold local file paths for the training, validation, and test folders. 
# FIXME: Directories will need to be changed for users not on this machine.
training_folder = r'C:\Users\Ankit Bombwal\Python Machine Learning\train'
validation_folder = r'C:\Users\Ankit Bombwal\Python Machine Learning\valid'
test_folder = r'C:\Users\Ankit Bombwal\Python Machine Learning\test'

# Converts image files to be consistent for Neural Network to function
# resizes to 128x128p and changes image file to a PyTorch Tensor.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Variables for each data folder. Holds object of the PlayingCardDataset class
# Parameters hold the file directory as well as the transformed Tensor data.
training_dataset = PlayingCardDataset(training_folder, transform)
validation_dataset = PlayingCardDataset(validation_folder, transform)
test_dataset = PlayingCardDataset(test_folder, transform)

# Creating our Data Loaders. This will load the training data in batches for better speed during runtime
# Data Loader wraps the data so that we can feed it into the Neural Network.
# Inputs are the cleaned data from dataset, and will be loaded in randomized batches of 32 items.
trainLoader = DataLoader(training_dataset, batch_size = 32, shuffle = True)
validateLoader = DataLoader(validation_dataset, batch_size = 32, shuffle = False)
testLoader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# Checks if GPU is available for Training and Validation
# We want cuda:0 as our output not CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
print(device)

# Object for the CardClassifier class
model = CardClassifier(num_classes=53)
# Changes the model to use the GPU instead of CPU
model.to(device)                            

# Loss function for the training loop
criteria = nn.CrossEntropyLoss()

# Optimizer for Deep Learning. Takes parameters from CardClassifier class and sets the learning rate at 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# FIXME: epochs are the number of times we will be looping through the whole dataset.
# empty list train_loss and val_loss are for holding both training and validation losses. 
# losses are a metric that describes how wrong a model's predictions are. We want low numbers.
epochs = 5
train_losses, val_losses = [], []

for epoch in range(epochs):
    # Training Phase
    model.train()
    running_loss = 0.0  # Records the loss data for current run

    # Training Loop runs through the loaded batches in trainLoader
    for images, labels in tqdm(trainLoader, desc='Training Loop'):
        images, labels = images.to(device), labels.to(device)   # Changes device to GPU
        optimizer.zero_grad()                                   # Resets the gradients of all parameters tracked by the optimizer at the start of the loop
        outputs = model(images)                                 # Calls the forward method in the CardClassifier class using the model class object
        loss = criteria(outputs, labels)                        # Calculates the loss for the current batch of tensors
        loss.backward()                                         # Back Propagation to update the model's weights 
        optimizer.step()                                        # step method call to adjust parameters based on updated weights
        running_loss += loss.item() * images.size(0)            # Adds together running loss from each loop in the training loop. Used to calculate training loss later
    
    # Calculating training loss and appending to the list for tracking
    train_loss = running_loss / len(trainLoader)
    train_losses.append(train_loss)

    # Validation Phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(validateLoader, desc='Validation Loop'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteria(outputs, labels)
            running_loss += loss.item() * images.size(0)

    # Calculating validation loss and appending to the list for tracking
    val_loss = running_loss / len(validateLoader)
    val_losses.append(val_loss)

    # Prints the stats for the current epoch
    print(f"Epoch {epoch+1}/{epochs}  - Train Loss: {train_loss}, Validation Loss: {val_loss} \n")

# Visualizing Training and Validation Loss Data
plt.plot(train_losses, label = 'Training Loss',)
plt.plot(val_losses, label = 'Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()

# Load and preprocess single image file for test case
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# Predicts using the model for a single image tensor
# Returns a 1-dimensional array containing the probabilities generated by the model for each image_tensor
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Visualization of the Model predictions
# Outputs the original image and a bar graph of percentages of what the model thinks each card is.
def visual(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Displays image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Displays Predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel('Probability')
    axarr[1].set_title('Class Predictions')
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

# Program now trains Pytorch model for image classification of playing cards.
# Outputs runtime data in visual graphs and charts for easier understanding.

# Variable to hold the number of correct predictions the model has made
correct_predictions = 0

# Number of samples taken from the test dataset
num_samples = 10

# Randomly select num_samples images from within the test file directory for the purposes of testing the model on more than one input
test_images = glob(r'C:\Users\Ankit Bombwal\Python Machine Learning\test\*\*')
test_examples = np.random.choice(test_images, num_samples)

# Loop to output model predictions from the test dataset.
for example in test_examples:
    original_image, image_tensor = preprocess_image(example, transform)
    probabilities = predict(model, image_tensor, device)
    class_names = test_dataset.classes
    visual(original_image, probabilities, class_names)

    # Determine the predicted class and the correct class
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]

    # Extract the true label from the file path (assuming folder structure reflects labels)
    true_label = example.split('\\')[-2]

    # Check if the prediction is correct
    if predicted_class == true_label:
        correct_predictions += 1

# Calculates and outputs the percent accuracy of the model's predictions 
percentage_correct = (correct_predictions / num_samples) * 100
print(f'The Card Classification model is {round(percentage_correct, 2)}% accurate at predicting the dataset after {epochs} training and validation epochs')