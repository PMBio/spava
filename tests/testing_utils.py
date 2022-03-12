import sys


def is_debug():
    if sys.gettrace() is not None:
        return True
    else:
        return False
