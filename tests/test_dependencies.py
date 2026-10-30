import subprocess
import sys


def test_testclient_import_has_no_deprecated_dependency_access():
    subprocess.run(
        [sys.executable, "-Werror", "-c", "from fastapi.testclient import TestClient"],
        check=True,
        capture_output=True,
        text=True,
    )
