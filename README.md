## Environment Setup
This project was originally developed using **Python 3.11** and the **uv** package manager for fast dependency management.  
Equivalent setup instructions using **pip** or **conda** are also provided for users who prefer those tools.

The project requires:
```toml
Python ~=3.11.0
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
It is recommended to reate a virtual environment to not interfere with your global installation.

```bash
uv venv --python 3.11.15
```

### 3. Activate your virtual environment

```bash
source .venv/bin/activate
```

### 4. Initialize project (if needed)

```bash
uv init
```

### 5. Confirm Python requirement in pyproject.toml file

```toml
requires-python = "~=3.11.0"
```

### 6. Install dependencies using lockfile

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