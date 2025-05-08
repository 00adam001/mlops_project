import gdown
import os

def download_from_gdrive(file_path='data/raw/sudoku.csv', file_id='12c_UTy7pXdzJkuL1HfVdaTP15Z8QZgA2'):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print("Downloading dataset from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    else:
        print("File already exists:", file_path)

if __name__ == "__main__":
    download_from_gdrive()
