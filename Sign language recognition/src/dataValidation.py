import os
import shutil

# Define the original data directory
original_data_dir = 'Data/Final Data'

# Define the new directories for training and validation data
train_dir = 'Data/Final_Training'
val_dir = 'Data/Final_Validation'

# Define the ratio of the split
split_ratio = 0.7

# Create the new directories if they don't exist
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

# Loop through each class directory in the original data directory
for class_dir in os.listdir(original_data_dir):
    if class_dir == ".DS_Store":
        continue
    class_path = os.path.join(original_data_dir, class_dir)

    # Create a new subdirectory in the training and validation directories for this class
    train_class_dir = os.path.join(train_dir, class_dir)
    val_class_dir = os.path.join(val_dir, class_dir)
    if not os.path.exists(train_class_dir):
        os.mkdir(train_class_dir)
    if not os.path.exists(val_class_dir):
        os.mkdir(val_class_dir)

    # Loop through each image file in the class directory and move it to either the training or validation directory
    for i, file_name in enumerate(os.listdir(class_path)):
        file_path = os.path.join(class_path, file_name)
        if i < split_ratio * len(os.listdir(class_path)):
            shutil.copy(file_path, os.path.join(train_class_dir, file_name))
        else:
            shutil.copy(file_path, os.path.join(val_class_dir, file_name))
