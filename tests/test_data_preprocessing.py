import sys
import subprocess

def test_data_preprocessing():
    import os
    os.environ["CI_TEST"] = "aaa"
    subprocess.run('python data2.py', shell=True, stdout=sys.stdout, stderr=sys.stderr, check=True)
    # there should be a "import data2" here, be careful that PyCharm does not remove it
    import data2


if __name__ == "__main__":
    test_data_preprocessing()
