# Envs
| name                                                                                                                                                         | type |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![Setup env](https://github.com/PMBio/a/actions/workflows/setup_env.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/setup_env.yaml)           | on demand |
| [![Setup env](https://github.com/PMBio/a/actions/workflows/setup_env_scvi.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/setup_env_scvi.yaml) | on demand |
# Tests
| name | type |
|-------------|---------|
|[![Run tests short](https://github.com/PMBio/a/actions/workflows/run_tests_short.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/run_tests_short.yaml)| on demand |
|[![Run tests](https://github.com/PMBio/a/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/run_tests.yaml)| on demand |

# Analyses
## IMC Jackson-Fischer dataset
| name                                                                                                                                                                                                                                    | type |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![Preprocess](https://github.com/PMBio/a/actions/workflows/imc_preprocess.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_preprocess.yaml)                                                                           | on demand |
| [![Tiler](https://github.com/PMBio/a/actions/workflows/imc_tiler.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_tiler.yaml)                                                                                          | on demand (no Juptyer)|
| [![Graphs](https://github.com/PMBio/a/actions/workflows/imc_graphs.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_graphs.yaml)                                                                                       | on demand |
| [![Loaders](https://github.com/PMBio/a/actions/workflows/imc_loaders.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_loaders.yaml)                                                                                    | on demand |
| `Train expression VAE`                                                                                                                                                                                                             | no workflow |
| [![Analyze expression VAE](https://github.com/PMBio/a/actions/workflows/imc_analyze_expression_vae.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_analyze_expression_vae.yaml)                                       | on demand |
| `Train image expression VAE`                                                                                                                                                                                                       | no workflow |
| [![Analyze image expression VAE](https://github.com/PMBio/a/actions/workflows/imc_analyze_image_expression_vae.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_analyze_image_expression_vae.yaml)                     | on demand |
| `Train image expression PCA VAE`                                                                                                                                                                                                       | no workflow |
| [![Analyze image expression PCA VAE](https://github.com/PMBio/a/actions/workflows/imc_analyze_image_expression_pca_vae.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_analyze_image_expression_pca_vae.yaml)                     | on demand |
| `Train image expression conv VAE`                                                                                                                                                                                             | no workflow |
| [![Analyze image expression conv VAE](https://github.com/PMBio/a/actions/workflows/imc_analyze_image_expression_conv_vae.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_analyze_image_expression_conv_vae.yaml) | on demand |

## IMC Jeongbin dataset
| name | type |
|-----------------------------------------------------------------------|---------|
| [![Preprocess](https://github.com/PMBio/a/actions/workflows/jeongbin_imc_preprocess.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/jeongbin_imc_preprocess.yaml) | on demand |

## Visium mouse brain dataset
| name                                                                                                                                                                                                                                                                                                                                                | type |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![Preprocess](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_preprocess.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_preprocess.yaml)                                                                                                                                                           | on demand |
| [![Tiler 32](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_tiler32.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_tiler32.yaml)                                                                                                                                                                          | on demand |
| [![Tiler 64](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_tiler64.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_tiler64.yaml)                                                                                                                                                                          | on demand |
| [![Graphs](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_graphs.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_graphs.yaml)                                                                                                                                                                       | on demand |
| [![Loaders](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_loaders.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_loaders.yaml)                                                                                                                                                                    | on demand |
| `Train expression VAE`                                                                                                                                                                                                                                                                                                                              | no workflow |
| [![Analyze expression VAE](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_expression_vae.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_expression_vae.yaml)                                                                                                                       | on demand |
| `Train image expression conv VAE 32`                                                                                                                                                                                                                                                                                                         | no workflow |
| [![Analyze image expression conv VAE 32](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_image_expression_conv_vae32.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_image_expression_conv_vae32.yaml) | on demand |
| `Train image expression conv VAE 64`                                                                                                                                                                                                                                                                                                         | no workflow |
| [![Analyze image expression conv VAE 64](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_image_expression_conv_vae64.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_image_expression_conv_vae64.yaml) | on demand |

## Visium endometrium dataset
| name                                                                                                                                                                                                                                                                                                                                                | type |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![Preprocess](https://github.com/PMBio/a/actions/workflows/visium_endometrium_preprocess.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_preprocess.yaml)                                                                                                                                                           | on demand |

| [![Tiler 32](https://github.com/PMBio/a/actions/workflows/visium_endometrium_tiler32.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_tiler32.yaml)                                                                                                                                                                          | on demand |
| [![Tiler large](https://github.com/PMBio/a/actions/workflows/visium_endometrium_tiler_large.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_tiler_large.yaml)                                                                                                                                                                          | on demand |
| [![Loaders](https://github.com/PMBio/a/actions/workflows/visium_endometrium_loaders.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_loaders.yaml)                                                                                                                                                                    | on demand |
<!---
| [![Graphs](https://github.com/PMBio/a/actions/workflows/visium_endometrium_graphs.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_graphs.yaml)                                                                                                                                                                       | on demand |
--->
<!---
| `Train expression VAE`                                                                                                                                                                                                                                                                                                                              | no workflow |
| [![Analyze expression VAE](https://github.com/PMBio/a/actions/workflows/visium_endometrium_analyze_expression_vae.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_analyze_expression_vae.yaml)                                                                                                                       | on demand |
| `Train image expression conv VAE 32`                                                                                                                                                                                                                                                                                                         | no workflow |
| [![Analyze image expression conv VAE 32](https://github.com/PMBio/a/actions/workflows/visium_endometrium_analyze_image_expression_conv_vae32.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_analyze_image_expression_conv_vae32.yaml) | on demand |
| `Train image expression conv VAE 64`                                                                                                                                                                                                                                                                                                         | no workflow |
| [![Analyze image expression conv VAE 64](https://github.com/PMBio/a/actions/workflows/visium_endometrium_analyze_image_expression_conv_vae64.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_endometrium_analyze_image_expression_conv_vae64.yaml) | on demand | --->

### scvi
| name                                                                                                                                                     | type |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![IMC scvi](https://github.com/PMBio/a/actions/workflows/imc_scvi.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_scvi.yaml)          | on demand (model trained manually) |
| [![Visium mousebrain scvi](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_scvi.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_scvi.yaml) | on demand (model trained manually) |

## Visium endometrium dataset
| name | type |
|-----------------------------------------------------------------------|---------|
