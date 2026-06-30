#!/usr/bin/env python3
"""Command-line entry point for description embedding."""

import sys

from smt_select.representations.desc_encoder import main


if __name__ == "__main__":
    sys.exit(main())
