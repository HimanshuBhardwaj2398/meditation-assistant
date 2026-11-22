"""Light ingestion modules must not drag in the ML stack on import.

``core.interfaces`` once imported ``langchain.schema``, which loads
transformers and torch: about 3 s before a SuttaCentral ref could even be
parsed. Checked in a fresh interpreter, since the test process has already
imported everything.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_suttacentral_and_core_import_without_torch():
    code = (
        "import sys\n"
        "import core, ingestion, ingestion.suttacentral\n"
        "heavy = [m for m in ('torch', 'transformers', 'sentence_transformers') "
        "if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
