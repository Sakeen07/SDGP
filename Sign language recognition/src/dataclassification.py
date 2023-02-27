import os
import shutil

# Define the mapping from old classes to new classes
class_mapping = {
    "A": "First_set",
    "B": "Second_set",
    "C": "Third_set",
    "D": "Second_set",
    "E": "First_set",
    "F": "Second_set",
    "G": "Fourth_set",
    "H": "Fourth_set",
    "I": "Fifth_set",
    "J": "Sixth_set",
    "K": "Second_set",
    "L": "Second_set",
    "M": "First_set",
    "N": "First_set",
    "O": "Third_set",
    "P": "Seventh_set",
    "Q": "Seventh_set",
    "R": "Second_set",
    "S": "First_set",
    "T": "First_set",
    "U": "Second_set",
    "V": "Second_set",
    "W": "Second_set",
    "X": "Fifth_set",
    "Y": "Sixth_set",
    "Z": "Seventh_set"
}

# Define the paths to the input and output directories
input_dir = "Data/ASL"
output_dir = "Data/Final_ASL"

# Loop through all the images in the input directory
for filename in os.listdir(input_dir):
    # Skip any files that start with a dot (hidden files)
    if filename.startswith('.'):
        continue

    # Determine the old class label for the image based on the filename
    old_class = filename.split("_")[0]

    # Map the old class label to the new class label
    new_class = class_mapping[old_class]

    # Create the directory for the new class if it doesn't exist
    new_class_dir = os.path.join(output_dir, new_class)
    if not os.path.exists(new_class_dir):
        os.makedirs(new_class_dir)

    # Move the image to the appropriate directory
    old_path = os.path.join(input_dir, filename)
    new_path = os.path.join(new_class_dir, filename)
    shutil.move(old_path, new_path)
