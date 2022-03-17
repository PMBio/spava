#!/bin/bash
export SPATIALMUON_FLAGS="TILE_SIZE=64"
python -m analyses.visium_mousebrain.image_expression_conv_vae_runner
unset SPATIALMUON_FLAGS
