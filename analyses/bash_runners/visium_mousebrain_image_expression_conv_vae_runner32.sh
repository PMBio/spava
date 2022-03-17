#!/bin/bash
export SPATIALMUON_FLAGS="TILE_SIZE=32"
python -m analyses.visium_mousebrain.image_expression_conv_vae_runner
unset SPATIALMUON_FLAGS
