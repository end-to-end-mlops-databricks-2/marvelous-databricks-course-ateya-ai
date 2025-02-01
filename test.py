from pathlib import Path
import os


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
print("Package root: ", PACKAGE_ROOT)
