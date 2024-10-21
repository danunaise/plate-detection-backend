import pandas as pd
import os

try:
    # Define the folder path where the images are located
    folder_path = 'C:\\Users\\danun\\Downloads\\image plate'

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    # Get a list of all .jpg files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # If no JPG files are found, raise an error
    if not jpg_files:
        raise FileNotFoundError(f"No .jpg files found in the folder {folder_path}.")

    # Create a list of paths
    file_paths = [os.path.join(folder_path, f) for f in jpg_files]

    # Create a DataFrame with file paths and an empty 'label' column
    df = pd.DataFrame({
        'path': file_paths,
        'label': ''  # You will fill in the labels manually
    })

    # Define the CSV path
    csv_path = 'C:\\Users\\danun\\Downloads\\image plate\\plate.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    print(f"CSV file saved successfully to {csv_path}")

except Exception as e:
    print(f"An error occurred: {e}")
