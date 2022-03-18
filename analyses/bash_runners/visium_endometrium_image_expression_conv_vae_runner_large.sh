#!/bin/bash
export SPATIALMUON_FLAGS="TILE_SIZE=large,DATASET_NAME=visium_endometrium"
python -m analyses.visium.image_expression_conv_vae_runner
unset SPATIALMUON_FLAGS
