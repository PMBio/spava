def test_data_import():
    import os
    os.environ["CI_TEST"] = "aaa"
    # there should be a "import data2" here, be careful that PyCharm does not remove it
    import data2


if __name__ == "__main__":
    test_data_import()
