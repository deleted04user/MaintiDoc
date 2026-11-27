import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from functools import cached_property
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5
from zlib import crc32

from qdrant_client import QdrantClient, models as qm

from .corpus import DATA, MANUAL, extract

TEXT = "maintidoc_text"
FIGURES = "maintidoc_figures"
EXACT = re.compile(r"\b(?:FIG|AL|INC|DIAG-FC|DISC|PROC-SAN|ALARM-TAB)-\d+\b", re.I)
STOP_WORDS = set("de des du le la les un une et en est sur pour au aux que quel quelle dans".split())


class Encoders:
    signature = "minilm-multilingual/clip-b32/bge-m3-int8"

    @cached_property
    def text_model(self):
        from fastembed import TextEmbedding
        return TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", threads=2)

    @cached_property
    def image_model(self):
        from fastembed import ImageEmbedding
        return ImageEmbedding("Qdrant/clip-ViT-B-32-vision", threads=2)

    @cached_property
    def reranker(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        return TextCrossEncoder("BAAI/bge-reranker-v2-m3-int8", threads=2)

    def texts(self, values):
        return [item.tolist() for item in self.text_model.embed(values, batch_size=8)]

    def images(self, values):
        from PIL import Image
        values = [Image.open(BytesIO(value)).convert("RGB") if isinstance(value, bytes)
                  else value for value in values]
        return [item.tolist() for item in self.image_model.embed(values, batch_size=2)]

    def rank(self, query, values):
        return list(self.reranker.rerank(query, values, batch_size=4))


def lexical(text):
    plain = "".join(c for c in unicodedata.normalize("NFKD", text.lower())
                    if not unicodedata.combining(c))
    tokens = Counter(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", plain))
    weights = {crc32(word.encode()): count / (count + 0.5)
               for word, count in tokens.items() if word not in STOP_WORDS}
    return qm.SparseVector(indices=list(weights), values=list(weights.values()))


class Index:
    def __init__(self, root=DATA, encoders=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.active = root / "active-index.json"
        self.encoders = encoders or Encoders()
        url = os.getenv("QDRANT_URL")
        self.client = (QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=15)
                       if url else QdrantClient(path=str(root / "qdrant"),
                                                force_disable_check_same_thread=True))

    def index(self, pdf=MANUAL):
        content = pdf.read_bytes()
        fingerprint = hashlib.sha256(content).hexdigest()
        previous = json.loads(self.active.read_text()) if self.active.exists() else {}
        if previous.get("sha256") == fingerprint and previous.get("encoder") == self.encoders.signature:
            return previous
        generation = uuid4().hex
        assets = self.root / "assets" / generation
        assets.mkdir(parents=True)
        snapshot = assets / "source.pdf"
        snapshot.write_bytes(content)
        rows = extract(snapshot, assets)
        for collection, figures in ((TEXT, False), (FIGURES, True)):
            items = [row for row in rows if (row["kind"] == "figure") == figures]
            if not items:
                continue
            dense = self.encoders.texts([f'{row["heading"]} {row["text"]}' for row in items])
            config = {"semantic": qm.VectorParams(size=len(dense[0]), distance=qm.Distance.COSINE)}
            visual = self.encoders.images([assets / row["image"] for row in items]) if figures else []
            if figures:
                config["visual"] = qm.VectorParams(size=len(visual[0]), distance=qm.Distance.COSINE)
            if not self.client.collection_exists(collection):
                self.client.create_collection(
                    collection, vectors_config=config,
                    sparse_vectors_config={"lexical": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
                )
            points = []
            for i, row in enumerate(items):
                vector = {"semantic": dense[i], "lexical": lexical(f'{row["element_id"]} {row["text"]}')}
                if figures:
                    vector["visual"] = visual[i]
                key = uuid5(NAMESPACE_URL, f'{generation}/{row["page"]}/{row["element_id"]}')
                points.append(qm.PointStruct(id=str(key), vector=vector, payload={
                    **row, "filename": pdf.name, "generation": generation, "sha256": fingerprint,
                }))
            self.client.upsert(collection, points=points, wait=True)
        result = {"generation": generation, "sha256": fingerprint,
                  "encoder": self.encoders.signature, "elements": len(rows),
                  "figures": sum(row["kind"] == "figure" for row in rows)}
        pending = self.root / f"{generation}.json"
        pending.write_text(json.dumps(result), encoding="utf-8")
        pending.replace(self.active)
        return result

    def search(self, query, photo=None, limit=6):
        manifest = json.loads(self.active.read_text(encoding="utf-8"))
        if manifest["encoder"] != self.encoders.signature:
            raise ValueError("Réindexez le manuel avec les encodeurs actifs.")
        scope = qm.Filter(must=[qm.FieldCondition(
            key="generation", match=qm.MatchValue(value=manifest["generation"]))])
        semantic = self.encoders.texts([query])[0]
        sparse = lexical(query)
        candidates = {}
        fusion = Counter()
        visual_match = None
        for collection in (TEXT, FIGURES):
            if not self.client.collection_exists(collection):
                continue
            queries = {"semantic": semantic}
            if sparse.indices:
                queries["lexical"] = sparse
            if collection == FIGURES and photo is not None:
                queries["visual"] = self.encoders.images([photo])[0]
            for name, vector in queries.items():
                hits = self.client.query_points(collection, query=vector, using=name,
                                                query_filter=scope, limit=8).points
                if name == "visual" and hits:
                    visual_match = hits[0].id
                for rank, hit in enumerate(hits, 1):
                    candidates[hit.id] = hit.payload
                    fusion[hit.id] += 1 / (20 + rank)
        if not candidates:
            return []
        codes = set(EXACT.findall(query.upper()))

        def exact(key):
            row = candidates[key]
            return int(row["element_id"] in codes or
                       ("ALARM-TAB-01" in codes and row["element_id"].startswith("AL-")))

        shortlist = sorted(candidates, key=lambda key: (exact(key), fusion[key]), reverse=True)[:14]
        values = [f'{candidates[key]["heading"]} {candidates[key]["text"]}' for key in shortlist]
        scores = dict(zip(shortlist, self.encoders.rank(query, values)))
        ranked = sorted(shortlist, key=lambda key: (exact(key), scores[key], fusion[key]), reverse=True)
        selected = ranked[:limit]
        if visual_match is not None:
            # The text reranker cannot compare pixels. Preserve the nearest image as evidence.
            ordered = [visual_match] + [key for key in ranked if key != visual_match]
            selected = sorted(ordered, key=exact, reverse=True)[:limit]
        figure = next((key for key in ranked if candidates[key]["kind"] == "figure"), None)
        if figure and figure not in selected and (photo is not None or re.search(r"figure|schéma|schema|diagram", query, re.I)):
            selected[-1] = figure
        if os.getenv("MAINTIDOC_LOW_MEMORY") == "1":
            for model in ("text_model", "image_model", "reranker"):
                self.encoders.__dict__.pop(model, None)
        return [candidates[key] for key in selected]

    def image_path(self, row):
        return self.root / "assets" / row["generation"] / row["image"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indexe le manuel SAN dans Qdrant.")
    parser.add_argument("pdf", type=Path, nargs="?", default=MANUAL)
    args = parser.parse_args()
    index = Index()
    try:
        print(json.dumps(index.index(args.pdf), indent=2))
    finally:
        index.client.close()
