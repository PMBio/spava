def test_preprocess_imc_data():
    import os

    os.environ["CI_TEST"] = "aaa"

    # there should be a "import imc_data" here, be careful that PyCharm does not remove it
    import imc_data


if __name__ == "__main__":
    test_preprocess_imc_data()
