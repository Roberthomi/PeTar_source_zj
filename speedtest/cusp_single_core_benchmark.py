#!/usr/bin/env python3
"""Benchmark PeTar with an isotropic Hernquist central density cusp."""

from plummer_single_core_benchmark import main


if __name__ == "__main__":
    main(profile="hernquist")
