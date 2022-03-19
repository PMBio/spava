#!/bin/bash
export SPATIALMUON_NOTEBOOK="analyses/visium_endometrium/expression_vae_analysis.py"
export SPATIALMUON_FLAGS="TILE_SIZE=large,MODEL_NAME=image_expression_conv_vae"
python notebook_exporter.py analyses/visium_endometrium/expression_vae_analysis.py ci_env
unset SPATIALMUON_FLAGS