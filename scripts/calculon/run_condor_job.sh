#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e 
set -x
#Arguments
REQUIREMENTS_FILE=$1  # Path to requirements.txt
# Shift the first argument to get the remaining arguments as $@
shift 1
COMMAND=$@

# Directories
ENV_CACHE_DIR="/mnt/scratch/users/gbarbadillo/arc25/cached-environments"  # Directory to cache environments
ENV_HASH=$(md5sum $REQUIREMENTS_FILE | awk '{print $1}')  # Hash the requirements.txt
ENV_DIR="$ENV_CACHE_DIR/venv_$ENV_HASH"

# Check if the environment exists
if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists. Reusing... $ENV_DIR"
    source $ENV_DIR/bin/activate
else
    echo "Environment does not exist. Creating a new one..."
    python3 -m venv $ENV_DIR 
    source $ENV_DIR/bin/activate
    pip3 install --upgrade pip
    pip3 install -r $REQUIREMENTS_FILE
    MAX_JOBS=2 pip3 install flash-attn==2.6.3 --no-build-isolation
    echo "Environment created and cached at $ENV_DIR."
fi

$COMMAND
deactivate
