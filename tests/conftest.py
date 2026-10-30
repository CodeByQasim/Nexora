import os
import sys

# Ensure active Python environment's Scripts (or bin) directory is on PATH
# so that test runners (pytest, etc.) resolve to this virtualenv rather than
# any system/base Python with incompatible packages.
_scripts_dir = os.path.join(sys.prefix, "Scripts" if os.name == "nt" else "bin")
if os.path.isdir(_scripts_dir):
    current_path = os.environ.get("PATH", "")
    if _scripts_dir.lower() not in current_path.lower():
        os.environ["PATH"] = _scripts_dir + os.pathsep + current_path
