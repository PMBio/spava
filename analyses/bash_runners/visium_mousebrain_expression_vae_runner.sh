#!/bin/bash
unset SPATIALMUON_FLAGS
python -m analyses.visium_mousebrain.expression_vae_runner
unset SPATIALMUON_FLAGS
