import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["MAINTIDOC_DATA"] = str(ROOT / ".maintidoc" / "preview")
os.environ["HF_HUB_OFFLINE"] = "1"

from maintidoc import retrieval


class PreviewEncoders(retrieval.Encoders):
    signature = "local-preview-minilm"

    def images(self, values):
        # Required by the figure collection schema. Preview never queries this vector.
        return [[1.0] for _ in values]

    def rank(self, query, values):
        return [0.0 for _ in values]


class PreviewIndex(retrieval.Index):
    def __init__(self):
        super().__init__(encoders=PreviewEncoders())

    def search(self, query, photo=None, limit=6):
        return super().search(query, limit=limit)


if __name__ == "__main__" and "--index" in sys.argv:
    index = PreviewIndex()
    try:
        print(index.index(), flush=True)
        top = index.search("Que signifie AL-01 ?")[0]
        assert (top["element_id"], top["page"]) == ("AL-01", 5)
        print(f'{top["element_id"]}, p. {top["page"]} | {top["text"]}', flush=True)
    finally:
        index.client.close()
elif __name__ == "__main__":
    import streamlit as st

    st.sidebar.info("Aperçu local. Recherche textuelle et OCR. Similarité visuelle et reclassement désactivés.")
    runpy.run_path(str(ROOT / "app.py"), init_globals={"index_factory": PreviewIndex},
                   run_name="__main__")

