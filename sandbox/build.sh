#!/bin/sh
# Build the sandbox image (the only step with network access) and record its image digest,
# which src/rewardhack/task.py checks against the local tag and stores in every Sample's metadata.
# Attestations are off because BuildKit's default provenance record changes the image ID on every
# build, even a fully cached one; without them a cached rebuild keeps the same ID.
# The ID is still per machine: an uncached build gives new layers and a new ID, and this script
# then rewrites the tracked image_digest.txt.
set -eu
cd "$(dirname "$0")"
docker build --provenance=false --sbom=false -t rewardhack-sandbox:latest .
docker image inspect --format '{{.Id}}' rewardhack-sandbox:latest > image_digest.txt
cat image_digest.txt
