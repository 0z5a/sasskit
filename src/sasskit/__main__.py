"""sasskit CLI entry point: python -m sasskit <command> [args]"""
import sys
from sasskit.recolor.cli import main

if __name__ == "__main__":
    sys.exit(main())
