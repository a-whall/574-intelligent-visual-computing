# Getting Started:

### 0. Navigate to the assignment-2/ directory.
- `cd 574-intelligent-visual-computing/assignment-2`

### 1. This assignment requires Python version >= 3.7.
- [Official Download](https://www.python.org/downloads/)

### 2. Create a new Python virtual environment.
- (On Unix or MacOS): `python3 -m venv .env`
- (On Windows): `py -m venv .env`

### 3. Activate the virtual environment.
- (On Unix or MacOS): `source .env/bin/activate`
- (On Windows): `.env\Scripts\activate`

### 4. Install PyTorch.
- The version of PyTorch is dependent on system hardware, thus before installing the other dependencies, go to the PyTorch [official site](https://pytorch.org/get-started/locally/#start-locally) to get the appropriate command.

### 5. Install the remaining dependencies.
- `pip install -r requirements.txt`

    Note:
    
    In case vscode doesn't automatically detect the virtual environment, it will give warnings that some imports are unresolved. The code will still run, but to fix this, select the interpreter version in the bottom right of vscode (while the file is open), and change it to the python executable file in the `bin` or `Scripts` folder (found in `assignment-2/.env/`).

### 6. Download the dataset from huggingface hub.
- Since the dataset is private, an [access token](https://huggingface.co/docs/hub/security-tokens) is needed to download the dataset with Python.
- Once you have generated a `read` access token, copy and paste it into the provided script `get-dataset.py`.
- After that's done, simply run the script and you should see the dataset directory appear and at this point the project should be runnable.

---

# How to Run:

To install the dataset:
- (On Unix or MacOS): `python3 get-dataset.py`
- (On Windows): `py get-dataset.py`

To run the model:
- (On Unix or MacOS): `python3 run.py`
- (On Windows): `py run.py`

See `run.py` for how to load a pretrained model.

Note: After the first run, the dataset will be cached in numpy gzip files. To use this data, you must uncomment line 43 of `trainMVShapeClassifier.py`.