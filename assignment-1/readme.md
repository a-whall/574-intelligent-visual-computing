# Getting Started:

### 0. Navigate to the assignment-1/ directory.
- `cd 574-intelligent-visual-computing/assignment-1`

### 1. Ensure Python version 3.10 is installed on your system.
- [Official Download](https://www.python.org/downloads/)

### 2. Create a new Python virtual environment with Python 3.10.
- (On Unix or MacOS): `python3.10 -m venv venv`'
- (On Windows): `py -3.10 -m venv venv`

### 3. Activate the virtual environment.
- (On Unix or MacOS): `source venv/bin/activate`
- (On Windows): `.\venv\Scripts\activate`

### 4. Install the necessary Python packages.
- `pip install -r requirements.txt`

---

Note: In case vscode does not automatically detect the virtual environment, the imports will give warnings that the imports are unresolved. The code will still run, but to fix this, select the interpreter version in the bottom right of vscode (while the file is open), select the python executable file in the `bin` or `Script` folder (found in `assignment-1/venv/`).

---

# How to Run:

- (On Unix or MacOS): `py basicReconstruction.py --file=sphere.pts --method=mls`
- (On Windows): `py basicReconstruction.py --file=sphere.pts --method=mls`

Note: method may be `naive` or `mls`