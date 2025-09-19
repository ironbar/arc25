#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e
set -x
# Arguments
REQUIREMENTS_FILE=$1  # Path to requirements.txt
# Shift the first argument to get the remaining arguments as $@
shift 1
COMMAND=$@

# Directories / cache
ENV_CACHE_DIR="/mnt/scratch/users/gbarbadillo/arc25/cached-environments"
ENV_HASH=$(md5sum "$REQUIREMENTS_FILE" | awk '{print $1}')
ENV_TAR="$ENV_CACHE_DIR/venv_${ENV_HASH}.tgz"
[ -z "$TMPDIR" ] && { echo "TMPDIR environment variable is mandatory. (Typically set by Condor)";  exit 1; }
LOCAL_ENV_DIR="$TMPDIR/venv_${ENV_HASH}"
STABLE_ANCHOR="/tmp/.venv_${ENV_HASH}"
mkdir -p "$ENV_CACHE_DIR"

log() {
    echo "$(date): $1" >&2
}

log "Started job"
# Load from cache if available, else create locally and cache as tar
if [ -f "$ENV_TAR" ]; then
    log "Environment cache found. Extracting to $LOCAL_ENV_DIR ..."
    rm -rf "$LOCAL_ENV_DIR"
    mkdir -p "$LOCAL_ENV_DIR"
    tar -xzf "$ENV_TAR" -C "$LOCAL_ENV_DIR"
    ln -sfn "$LOCAL_ENV_DIR" "$STABLE_ANCHOR"
    # Activate via stable anchor so internal paths remain valid
    source "$STABLE_ANCHOR/bin/activate"
else
    log "Environment cache not found. Creating a new one using stable anchor $STABLE_ANCHOR ..."
    # Create at stable anchor so scripts/shebangs point to it
    python3 -m venv "$STABLE_ANCHOR"
    # Move to local storage and re-anchor
    mv "$STABLE_ANCHOR" "$LOCAL_ENV_DIR"
    ln -sfn "$LOCAL_ENV_DIR" "$STABLE_ANCHOR"

    # Install via the anchor's python to keep shebangs consistent
    "$STABLE_ANCHOR/bin/python" -m pip install --upgrade pip
    "$STABLE_ANCHOR/bin/python" -m pip install -r "$REQUIREMENTS_FILE"
    MAX_JOBS=2 "$STABLE_ANCHOR/bin/python" -m pip install flash-attn==2.6.3 --no-build-isolation

    # fix torch dataloader (detect python version inside venv)
    PYVER=$("$STABLE_ANCHOR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    DL_PATH="$LOCAL_ENV_DIR/lib/python$PYVER/site-packages/torch/utils/data/dataloader.py"
    if [ -f "$DL_PATH" ]; then
      sed -i.bak "0,/multiprocessing_context[[:space:]]*=[[:space:]]*None,/s//multiprocessing_context='\''fork'\'',/" "$DL_PATH"
    fi

    log "Creating environment cache tar at $ENV_TAR ..."
    tar -czf "$ENV_TAR" -C "$LOCAL_ENV_DIR" .

    # Activate via stable anchor
    source "$STABLE_ANCHOR/bin/activate"
fi
log "Environment ready"

$COMMAND
deactivate
log "Finished job"
