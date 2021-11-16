import os

os.chdir('/data/l989o/deployed/a')
os.environ["CI_TEST"] = "aaa"


def test_import_data():
    import data2


if __name__ == "__main__":
    test_import_data()
