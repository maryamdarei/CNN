
# CNN Model With PyTorch For Image Classification
# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import glob
import cv2

# Data Augmentation:
# flip horizontally and vertically, Resize to 128X128

def main():
    path_of_the_directory = "/home/azadeh/Projects/CNN/Data/small-train"
    for filename in os.listdir(path_of_the_directory):
        img_dir = os.path.join(path_of_the_directory, filename)
        jpg_ext = ".jpg"
        path = "/home/azadeh/Projects/CNN/Data/small-train"
        angles = [0]
        for file_name in glob.iglob(os.path.join(img_dir, "*" + jpg_ext)):
            original_img = Image.open(file_name)
            new_image = original_img.resize((128, 128))
            for angel in angles:
                rot_suffix = "_r{:03d}{:s}".format(angel, jpg_ext)
                file_name_rot = file_name.replace(jpg_ext, rot_suffix)
                vertical_img = new_image.transpose(method=Image.FLIP_TOP_BOTTOM)
                vertical_img.save(file_name_rot)
                print("Rotated: {:s} by {:3d} degrees to {:s}".format(file_name, angel, file_name_rot))

        angles = [1]
        for file_name in glob.iglob(os.path.join(img_dir, "*" + jpg_ext)):
            original_img = Image.open(file_name)
            new_image = original_img.resize((128, 128))
            for angel in angles:
                rot_suffix = "_r{:03d}{:s}".format(angel, jpg_ext)
                file_name_rot = file_name.replace(jpg_ext, rot_suffix)
                horz_img = new_image.transpose(method=Image.FLIP_LEFT_RIGHT)
                horz_img.save(file_name_rot)
                print("Rotated: {:s} by {:3d} degrees to {:s}".format(file_name, angel, file_name_rot))


if __name__ == "__main__":
    print("Python {:s} on {:s}\n".format(sys.version, sys.platform))
    main()
    print("\nDone.")


# Rotate 90 and 270:


def main():
    path_of_the_directory = "/home/azadeh/Projects/CNN/Data/small-train"
    for filename in os.listdir(path_of_the_directory):
        img_dir = os.path.join(path_of_the_directory, filename)
        jpg_ext = ".jpg"
        angles = [90]
        for file_name in glob.iglob(os.path.join(img_dir, "*" + jpg_ext)):
            # image = Image.open(file_name)
            original_img = Image.open(file_name)
            new_image = original_img.resize((128, 128))
            for angle in angles:
                rot_suffix = "_r{:03d}{:s}".format(angle, jpg_ext)
                file_name_rot = file_name.replace(jpg_ext, rot_suffix)
                image_rot = new_image.rotate(angle)
                image_rot.save(file_name_rot)
                print("Rotated: {:s} by {:3d} degrees to {:s}".format(file_name, angle, file_name_rot))


if __name__ == "__main__":
    print("Python {:s} on {:s}\n".format(sys.version, sys.platform))
    main()
    print("\nDone.")


# Transform Data:
# Define relevant variables for the ML task
batch_size = 64
num_classes = 683
learning_rate = 0.001
num_epochs = 20

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train and test data directory
data_dir = "/home/azadeh/Projects/CNN/Data/small-train"
test_data_dir = "/home/azadeh/Projects/CNN/Data/small-validation"



#load the train and test data
train_dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((128,128)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((128,128)),transforms.ToTensor()
]))

img, label = train_dataset[0]
print(img.shape,label)


# Display image:
def display_img(img,label):
    print(f"Label : {train_dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))

#display the first image in the dataset
display_img(*train_dataset[2])

# Splitting Data and Prepare Batches:
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

batch_size = 128
val_size = len(test_dataset)
train_size = len(train_dataset) - val_size

train_data,val_data = random_split(train_dataset,[train_size,val_size])
print(f"Length of Train Data : {len(train_dataset)}")
print(f"Length of Validation Data : {len(val_data)}")

#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)

# CNN from Scratch:
# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(53824, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Setting Hyperparameters
model = ConvNeuralNet(num_classes)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

total_step = len(train_dl)

# Trainning:
# We use the pre-defined number of epochs to determine how many iterations to train the network on
for epoch in range(num_epochs):
    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_dl):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Testing:
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_dl:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(34225, 100 * correct / total))

