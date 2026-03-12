## Environment Setup
This project was originally developed using **Python 3.11** and the **uv** package manager for fast dependency management.  
Equivalent setup instructions using **pip** or **conda** are also provided for users who prefer those tools.

The project requires:
```toml
requires-python = "~=3.11.0"
```
---

## Option 1: Using `uv` (recommended)

This is the primary environment workflow used for this project.

### 1. Ensure Python 3.11 is installed

```bash
python3 --version
```

Expected output:
```bash
Python 3.11.x
```
### 2. Create virtual environment
It is recommended to create a virtual environment so the project dependencies do not interfere with your global Python installation..

```bash
uv venv --python 3.11.15
```

### 3. Activate your virtual environment

On macOS and Linux:
```bash
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```

#### 3a. Confirm Python requirement in pyproject.toml file

```toml
requires-python = "~=3.11.0"
```

### 4. Install dependencies using lockfile

```bash
uv sync
```

This will install the exact dependency set recorded in uv.lock.

If the lockfile is not available and you need to recreate the environment manually, the original setup used:

```bash
uv add "pycaret[full]"
uv add esm
uv add ankh
```

## Option 2: Using `conda`

If you prefer conda, create an environment with Python 3.11 and install from requirements.txt.

### 1. Create the environment
```bash
conda create -n siads699-capstone python=3.11
```

### 2. Activate it
```bash
conda activate siads699-capstone
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Note: even inside a conda environment, pip is typically the easiest way to install this project’s Python dependencies from requirements.txt.

## Option 3: Using `pip`
If you prefer pip, you can create a standard virtual environment and install from requirements.txt.

### 1. Make sure Python 3.11 is installed

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
```

### 3. Activate the environment

On macOS and Linux:
```bash
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```