import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from huggingface_hub import hf_hub_download
from zipfile import ZipFile
from re import search
from shutil import rmtree

# Replace this value with your api-token as a string.
user_token=""

if os.path.isdir('dataset'):
    print("It seems like you've already downloaded the dataset.\nA directory named 'dataset' already exists.\nYou must delete the 'dataset' directory and then run this script again to re-download the dataset.")
    exit()

if user_token == "":
    print("Error: Edit this file to add your access token (see readme.md)")
    exit()

downloaded_path = hf_hub_download(
    repo_id="awhall/pointnet-registration",
    repo_type="dataset",
    filename="dataset.zip",
    cache_dir="./",
    use_auth_token=user_token
)

with ZipFile(downloaded_path, 'r') as zip:
    zip.extractall("./")

match = search(r"(\.\/[a-zA-Z0-9-_]+)[/\\]", downloaded_path)

if match is not None:
    root_hf_download_dir = match.group(1)
    rmtree(root_hf_download_dir)
else:
    print("Error: Couldn't delete the downloaded directory containing the zip file.")