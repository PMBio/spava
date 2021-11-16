# pytest -s
import subprocess
import sys


def test_envs():
    subprocess.run(
        "bash /data/l989o/deployed/a/tests/test_envs.sh",
        shell=True,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )

if __name__ == '__main__':
    test_envs()