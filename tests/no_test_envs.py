# pytest -s
import subprocess
import sys


def no_test_envs():
    subprocess.run(
        "bash /data/l989o/deployed/a/tests/no_test_envs.sh",
        shell=True,
        stderr=sys.stderr,
        stdout=sys.stdout,
        check=True
    )


if __name__ == "__main__":
    no_test_envs()
