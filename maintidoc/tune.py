"""Fit a small pairwise ranking head over the frozen multilingual encoder."""
import json
import math
from pathlib import Path

from .retrieval import Encoders, Index, lexical


# Independently worded training requests with a positive and a confusable negative.
TRIPLETS = [
    ("Le volume est inaccessible. Qui prévenir ?", "AL-03", "AL-04"),
    ("L’espace disponible devient insuffisant.", "AL-04", "AL-03"),
    ("Quelle alarme correspond à un disque en pré-alerte ?", "AL-02", "AL-01"),
    ("Le simulateur indique que tout est normal.", "AL-00", "AL-02"),
    ("Comment vérifier les ports après la disparition d’un chemin ?", "DIAG-FC-01", "DISC-02"),
    ("Quels contrôles effectuer sur le groupe de disques ?", "DISC-02", "DIAG-FC-01"),
    ("Le contrôleur B a présenté un événement intermittent.", "INC-130", "INC-117"),
    ("Quel incident mentionne une capacité de 88 % ?", "INC-125", "INC-104"),
]


def features(query, text, score):
    words = set(lexical(query).indices)
    return [math.tanh(float(score)/5), len(words.intersection(lexical(text).indices))/max(1,len(words))]


class AdaptedEncoders(Encoders):
    def rank(self, query, values):
        weights = json.loads(Path("ranking-head.json").read_text())["weights"]
        scores = super().rank(query, values)
        return [sum(w*x for w,x in zip(weights,features(query,text,score)))
                for text,score in zip(values,scores)]


def train(index):
    from qdrant_client import models as qm
    generation = json.loads(index.active.read_text())["generation"]
    rows, _ = index.client.scroll("maintidoc_text", limit=200, scroll_filter=qm.Filter(must=[
        qm.FieldCondition(key="generation", match=qm.MatchValue(value=generation))]))
    passages = {r.payload["element_id"]: r.payload["heading"]+" "+r.payload["text"] for r in rows}
    deltas = []
    for query, positive, negative in TRIPLETS:
        texts = [passages[positive], passages[negative]]
        scores = index.encoders.rank(query, texts)
        a,b = [features(query,t,s) for t,s in zip(texts,scores)]
        deltas.append([x-y for x,y in zip(a,b)])
    weights = [1.0, 0.0]
    for _ in range(100):
        for delta in deltas:
            scale = 0.05/(1+math.exp(sum(w*x for w,x in zip(weights,delta))))
            weights = [w+scale*x-0.0005*w for w,x in zip(weights,delta)]
    Path("ranking-head.json").write_text(json.dumps({"weights":weights,"training_pairs":len(deltas)},indent=2))


if __name__ == "__main__":
    index = Index()
    try:
        train(index)
    finally:
        index.client.close()
