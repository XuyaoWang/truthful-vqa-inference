## Getting Started

Follow these steps to set up and run the project:

### 1. Install `uv` Package Manager (if needed)

If you don't have `uv` installed, run the following command on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installation to ensure the `uv` command is available in your PATH.

### 2. Install the Project

Install the `truthful-vqa-inference` package and its dependencies using `uv`:

```bash
# Clone the repository
git clone git@github.com:XuyaoWang/truthful-vqa-inference.git

# Navigate into the project directory
cd truthful-vqa-inference

# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate

# Install the project dependencies
uv sync
```

### 3. Run

```bash
bash scripts/occlusion_high/gemma-3-27b-it.sh
```