# yes: python -m pytest -s
# no! pytest -s (problems with imports, if you are curious look here https://stackoverflow.com/questions/49028611/pytest-cannot-find-module)
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
