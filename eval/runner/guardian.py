"""Reap only this run's descendants if its coordinator disappears."""
import sys
import time
from .runner import cleanup, process_start

if __name__ == '__main__':
    pid, birth, token = sys.argv[1:]
    while process_start(int(pid)) == birth:
        time.sleep(2)
    cleanup(token)
