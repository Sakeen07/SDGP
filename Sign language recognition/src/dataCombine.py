import os
import shutil

# Set the source directories
source_dir1 = 'Data2/ASL/Z'
source_dir2 = 'Data2/BSL/Z'

# Set the destination directory
destination_dir = 'Data/Final Data/Z'

# If the destination directory doesn't exist, create it
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Loop through the files in the source directories and copy them to the destination directory
for file_name in os.listdir(source_dir1):
    file_path = os.path.join(source_dir1, file_name)
    shutil.copy(file_path, destination_dir)

for file_name in os.listdir(source_dir2):
    file_path = os.path.join(source_dir2, file_name)
    shutil.copy(file_path, destination_dir)