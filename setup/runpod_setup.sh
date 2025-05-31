#!/bin/bash
# adapted from: https://github.com/bongohead/interpretable-moes/blob/master/runpod_setup.sh
VENV_PATH="/workspace/.venv"

# check if .venv already exists in persistent storage 
if [ -d "$VENV_PATH" ]; then
    echo "✅ found existing virtual environment at $VENV_PATH"
    echo "To activate the virtual environment, run:"
    echo "  source $VENV_PATH/bin/activate"
else
    echo "🚧 no venv found. installing Python and setting up environment..."

    # Update and install all required packages at once
    apt update -y && apt upgrade -y
    apt install -y nano python3-pip python3.12 python3.12-dev python3.12-venv python3-distutils

    # set Python 3.12 as the default python
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

    # create virtual env (note the -m flag)
    python -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"

    # upgrade pip
    python -m pip install --upgrade pip
    python -m pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets
    python -m pip install ipykernel 
    python -m ipykernel install --user --name=python-3.12-venv --display-name="Python 3.12 (venv)"

    # install packages
    echo "$pwd"
    sh /workspace/adaptive-compute/setup/install.sh
fi 

echo "ready to experiment!"
