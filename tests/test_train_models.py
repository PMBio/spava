import os



def test_train_vae_expression():
    os.environ["SPATIALMUON_TEST"] = "analyses/vae_expression/vae_expression_model.py"
    import analyses.vae_expression.vae_expression_model


if __name__ == "__main__":
    test_train_vae_expression()
