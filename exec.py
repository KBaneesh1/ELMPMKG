import os
import zipfile

zip_file_name = 'fb15k_wn18_images.zip'  # Replace with the actual name of the zip file
destination_folder = 'dataset'

# Check if the destination folder exists
if not os.path.exists(destination_folder):
    print(f"The folder '{destination_folder}' does not exist. Please create it first.")
else:
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    print(f"Contents extracted to '{destination_folder}'")