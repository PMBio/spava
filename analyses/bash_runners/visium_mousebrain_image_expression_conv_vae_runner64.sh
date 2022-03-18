#!/bin/bash
export SPATIALMUON_FLAGS="TILE_SIZE=64,DATASET_NAME=visium_mousebrain"
python -m analyses.visium.image_expression_conv_vae_runner
unset SPATIALMUON_FLAGS
