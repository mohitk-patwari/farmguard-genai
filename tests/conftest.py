"""Tests never make noise: load_config forces alarm backend "none" while this is set."""
import os

os.environ["FARMGUARD_MUTE"] = "1"
