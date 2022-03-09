import os

os.environ["SPATIALMUON_TEST"] = "aaa"


def test_train_vae_expression():
    import analyses.vae_expression.vae_expression_model


if __name__ == "__main__":
    test_train_vae_expression()
