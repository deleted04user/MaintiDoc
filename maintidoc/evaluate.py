"""Measure retrieval on the twelve questions printed on page 7 of the manual."""
import json
import math
import re
import statistics
import time

import pymupdf
from .corpus import MANUAL
from .retrieval import Index

# Alternative acceptable sources within each group. Separate groups are complementary.
GOLD = [
    [{"FIG-01"}, {"FIG-02"}], [{"FIG-02"}], [{"LEXIQUE-FC-5"}],
    [{"INC-117"}, {"FIG-03"}], [{"AL-01"}], [{"P5-3"}],
    [{"PROC-SAN-01"}], [{"AUTORISATIONS-3", "PROC-SAN-01"}],
    [{"DIAG-FC-01"}], [{"AUTORISATIONS-4", "P6-4"}],
    [{"AL-04"}], [{f"EXEMPLE-TICKET-{n}" for n in range(1, 6)}],
]


def evaluate(index):
    with pymupdf.open(MANUAL) as doc:
        questions = [b[4].splitlines()[1] for b in doc[6].get_text("blocks", sort=True)
                     if re.match(r"Q\d\d\n", b[4])]
    if len(questions) != len(GOLD):
        raise ValueError("Les douze questions de référence sont requises.")
    cases = []
    for question, groups in zip(questions, GOLD):
        start = time.perf_counter()
        ids = [row["element_id"] for row in index.search(question, limit=10)]
        seen = set()
        dcg = 0
        ranks = []
        for rank, element in enumerate(ids, 1):
            for group, alternatives in enumerate(groups):
                if element in alternatives and group not in seen:
                    seen.add(group)
                    ranks.append(rank)
                    if rank <= 5:
                        dcg += 1 / math.log2(rank + 1)
        ideal = sum(1 / math.log2(i+2) for i in range(min(5, len(groups))))
        cases.append(dict(question=question, top5=ids[:5], seconds=round(time.perf_counter()-start, 2),
                          recall5=sum(r <= 5 for r in ranks)/len(groups), ndcg5=dcg/ideal,
                          mrr10=1/min(ranks) if ranks else 0))
    return {"encoder": index.encoders.signature, "cases": cases, "mean": {
        metric: round(statistics.mean(case[metric] for case in cases), 3)
        for metric in ("recall5", "ndcg5", "mrr10", "seconds")}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted", action="store_true", help="Utiliser la tête ajustée par maintidoc.tune")
    args = parser.parse_args()
    encoders = None
    if args.adapted:
        from .tune import AdaptedEncoders
        encoders = AdaptedEncoders()
    index = Index(encoders=encoders)
    try:
        print(json.dumps(evaluate(index), ensure_ascii=False, indent=2))
    finally:
        index.client.close()
