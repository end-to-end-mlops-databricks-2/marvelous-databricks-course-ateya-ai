import os

from titanic_survival_model.config import config

with open(os.path.join(config.PACKAGE_ROOT, "VERSION")) as f:
    __version__ = f.read().strip()
