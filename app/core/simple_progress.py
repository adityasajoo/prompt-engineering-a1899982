# app/core/simple_progress.py
# Zero-dependency, single-line percentage printer.

import sys

def print_pct(label: str, done: int, total: int):
    """Update a single-line percentage in the terminal.
    Call with done in [0..total]. Prints a newline only when you choose to.
    """
    if total <= 0:
        pct = 100
        done = total = 1
    else:
        pct = int(done * 100 / total)
    sys.stdout.write(f"\r{label}: {pct}% ({done}/{total})")
    sys.stdout.flush()

def newline():
    """Move to the next line (after the last percentage line)."""
    sys.stdout.write("\n")
    sys.stdout.flush()
