# Envs
| name                                                                                                                                                         | type |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![Setup env](https://github.com/PMBio/a/actions/workflows/setup_env.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/setup_env.yaml)           | on demand |
| [![Setup env](https://github.com/PMBio/a/actions/workflows/setup_env_scvi.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/setup_env_scvi.yaml) | on demand |
# Tests
| name | type |
|-------------|---------|
|[![Run tests](https://github.com/PMBio/a/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/run_tests.yaml)| on push |

# Analyses
## IMC Jackson-Fischer dataset
| name                                                                                                                                                                                                                                                                         | type |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![Preprocess](https://github.com/PMBio/a/actions/workflows/imc_preprocess.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_preprocess.yaml)                                                                                                                | on demand |
| [![Tiler](https://github.com/PMBio/a/actions/workflows/imc_tiler.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_tiler.yaml)                                                                                                                               | on demand (no Juptyer)|
| [![Graphs](https://github.com/PMBio/a/actions/workflows/imc_graphs.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_graphs.yaml)                                                                                                                            | on demand |
| [![Loaders](https://github.com/PMBio/a/actions/workflows/imc_loaders.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_loaders.yaml)                                                                                                                         | on demand |
| `Train expression VAE`                                                                                                         <br/>                                                                                                                                         | no workflow |
| [![Analyze expression VAE](https://github.com/PMBio/a/actions/workflows/imc_analyze_expression_vae.yaml/badge.svg)<br/><br/><br/><br/>]<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>(https://github.com/PMBio/a/actions/workflows/imc_analyze_expression_vae.yaml) | on demand |

## IMC Jeongbin dataset
| name | type |
|-----------------------------------------------------------------------|---------|
| [![Preprocess](https://github.com/PMBio/a/actions/workflows/jeongbin_imc_preprocess.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/jeongbin_imc_preprocess.yaml) | on demand |

## Visium mouse brain dataset
| name                                                                                                                                                                                                                                             | type |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![Preprocess](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_preprocess.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_preprocess.yaml)                                                        | on demand |
| [![Tiler](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_tiler.yaml/badge.svg)](https://github.<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>com/PMBio/a/actions/workflows/visium_mousebrain_tiler.yaml) | on demand |
| [![Graphs](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_graphs.yaml/badge.svg)](https://github.<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>com/PMBio/a/actions/workflows/visium_mousebrain_graphs.yaml)             | on demand |
| [![Loaders](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_loaders.yaml/badge.svg)](https://github.<br/>com/PMBio/a/actions/workflows/visium_mousebrain_loaders.yaml)                                                            | on demand |
| `Train expression VAE`                                                                                                                                                                                                                           | no workflow |
| [![Analyze expression](https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_expression_vae.yaml/badge.svg)]<br/><br/><br/>(https://github.com/PMBio/a/actions/workflows/visium_mousebrain_analyze_expression_vae.yaml)         | on demand |

### scvi
| name                                                                                                                                                     | type |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [![IMC scvi](https://github.com/PMBio/a/actions/workflows/imc_scvi.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/imc_scvi.yaml)          | on demand (model trained manually) |
| [![Visium scvi](https://github.com/PMBio/a/actions/workflows/visium_scvi.yaml/badge.svg)](https://github.com/PMBio/a/actions/workflows/visium_scvi.yaml) | on demand (model trained manually) |

## Visium endometrium dataset
| name | type |
|-----------------------------------------------------------------------|---------|
