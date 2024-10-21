import os

# Define the folder path where the images are located
folder_path = 'C:\\Users\\danun\\Downloads\\image plate'

# Get a list of all .jpg files in the folder
jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]


# Function to find a unique file name if it already exists
def get_unique_filename(folder, base_name, ext):
    i = 1
    new_name = f"{base_name}{ext}"
    while os.path.exists(os.path.join(folder, new_name)):
        new_name = f"{base_name}_{i}{ext}"
        i += 1
    return new_name


# Loop through each file and rename it
for i, old_name in enumerate(jpg_files, 1):
    # Create the base new name (plate1, plate2, etc.)
    base_new_name = f"plate{i}"
    ext = ".jpg"

    # Check if the new name already exists, and if so, get a unique name
    new_name = get_unique_filename(folder_path, base_new_name, ext)

    # Full paths for old and new names
    old_file_path = os.path.join(folder_path, old_name)
    new_file_path = os.path.join(folder_path, new_name)

    # Rename the file
    os.rename(old_file_path, new_file_path)
    print(f'Renamed: {old_name} -> {new_name}')

print("Renaming completed.")
