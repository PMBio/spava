import os

os.environ["CI_TEST"] = 'aaa'


def test_import_data():
    import data2
