import os
import errno
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from argparse import ArgumentParser
from getpass import getpass
from huggingface_hub import hf_hub_download
from zipfile import ZipFile
from re import search
from shutil import rmtree

# Various functions used during training/testing for use as a module.

# Alternatively, run this script directly for file-management utilities.

# Args:

# --get-dataset : Download the dataset from huggingface.
#                 a valid access-token to download the dataset.
#                 (required while the repo is private)
#                 Replace this value with your api-token as a string.

# --clear : clear any of the auto-generated files in this 

# ----------------------------------------------------------------

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def isdir(dirname):
    return os.path.isdir(dirname)

def isfile(fname):
    return os.path.isfile(fname)

class AverageMeter(object):

    """ Computes and stores the average and current value. """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count



def fetch_main(user_token):

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



def clean_main():

    if isdir("./output"):
        rmtree("./output")
    
    if isdir("./checkpoints"):
        rmtree("./checkpoints")



if __name__ == "__main__":

    parser = ArgumentParser(description='utils')
    parser.add_argument("--fetch-dataset", action="store_true", help="Download the dataset from huggingface.")
    parser.add_argument("-c", "--clean", action="store_true", help="Remove model checkpoints and outputs.")
    
    args = parser.parse_args()

    if args.fetch_dataset:
        if isdir("dataset"):
            print("It seems like you've already downloaded the dataset.\nA directory named 'dataset' already exists.\nYou must delete the 'dataset' directory and then run this script again to re-download the dataset.")
            exit()
        access_token = getpass("Enter your huggingface access token: ")
        fetch_main(access_token)

    if args.clean:
        clean_main()
