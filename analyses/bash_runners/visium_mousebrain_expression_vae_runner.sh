#!/bin/bash

export SPATIALMUON_FLAGS="TILE_SIZE=32,DATASET_NAME=visium_mousebrain"
python -m analyses.visium.expression_vae_runner
unset SPATIALMUON_FLAGS
