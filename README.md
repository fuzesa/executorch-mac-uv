# Setup
1. Make sure you have `uv` installed  
You can check their website at https://docs.astral.sh/uv/getting-started/installation/, but personally I just prefer to install it via HomeBrew `brew install uv`

2. Clone the repo, the create a virtual environment inside the directory and activate it
```shell
git clone https://github.com/fuzesa/executorch-mac-uv.git
cd executorch-uv
uv venv --python 3.12
source .venv/bin/activate
```

3. Sync the environment
```shell
uv sync
```

4. Run the app
```shell
uv run main.py
```
