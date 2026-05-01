"""Backward-compatible re-exports from the land_productivity package.

All functions and classes have been moved into the land_productivity package.
This module exists so existing notebooks continue to work without changes:
    from functions import generate_report_period, BasicArgs, ...
"""
from land_productivity import *  # noqa: F401,F403
from land_productivity import __all__  # noqa: F401
