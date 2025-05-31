#!/bin/bash
# check for .venv
if [ ! -d "./../.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ./../.venv
fi

# activate the virtual environment
source ./../.venv/bin/activate

# run your existing installation script
./install.sh
