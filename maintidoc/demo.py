"""Exercise the real workflow using the twelve manual questions and simulated camera inputs."""
import json
import time
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageEnhance
from langgraph.types import Command

from .corpus import EQUIPMENT
from .evaluate import evaluate
from .maintenance import Maintenance
from .retrieval import Index
from .workflow import build_workflow


def run(index, destination):
    report = evaluate(index)
    graph = build_workflow(index, Maintenance())
    snapshot = json.loads(index.active.read_text())["generation"]
    for figure in ("FIG-02", "FIG-03", "FIG-04"):
        image = Image.open(index.root/"assets"/snapshot/f"{figure}.png").convert("RGB")
        image.thumbnail((640, 480))
        image = ImageEnhance.Contrast(image.rotate(3, fillcolor="white")).enhance(0.75)
        output = BytesIO()
        image.save(output, "JPEG", quality=55)
        ids = [r["element_id"] for r in index.search("Quel schéma correspond à cette image ?", output.getvalue())]
        report.setdefault("simulated_photos", []).append({"expected":figure,"retrieved":ids})
    photo = Path("evaluation-data/qlogic.jpg")
    if photo.exists():
        with Image.open(photo) as image:
            image.thumbnail((960,960))
            output = BytesIO()
            image.convert("RGB").save(output,"JPEG")
        state = graph.invoke({"question":"Ce voyant orange permet-il de déduire la panne ?",
                              "photo":output.getvalue(), "actor":"evaluation","equipment_id":EQUIPMENT},
                             {"configurable":{"thread_id":uuid4().hex}})
        report["hardware_photo"] = {"file":str(photo),"answer":state["answer"],
                                    "retrieved":[r["element_id"] for r in state["evidence"]]}
    with tempfile.TemporaryDirectory() as root:
        service = Maintenance(Path(root))
        tickets = build_workflow(index, service)
        config = {"configurable":{"thread_id":uuid4().hex}}
        pending = tickets.invoke({"question":"Prépare un ticket pour AL-01.", "actor":"evaluation",
                                  "equipment_id":EQUIPMENT},config)
        assert pending["__interrupt__"] and not service.read(EQUIPMENT,"evaluation",["get_open_tickets"])["get_open_tickets"]["tickets"]
        approved = tickets.invoke(Command(resume=True),config)
        retry = service.create("evaluation",pending["request_id"],pending["draft"])
        assert retry["already_created"] and retry["ticket_id"] == approved["ticket"]["ticket_id"]
        report["ticket_demo"] = {"paused":True,"approved":True,"duplicate_prevented":True}
    for case in report["cases"]:
        thread = uuid4().hex
        started = time.monotonic()
        state = graph.invoke({"question":case["question"],"actor":"evaluation","equipment_id":EQUIPMENT,
                              "role":"reader"}, {"configurable":{"thread_id":thread}})
        case["answer"] = state["answer"]
        case["cited_elements"] = [state["evidence"][claim["source"]-1]["element_id"]
                                  for claim in state["answer"]["claims"]]
        case["answer_seconds"] = round(time.monotonic()-started, 2)
        graph.checkpointer.delete_thread(thread)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(case["question"], case["answer_seconds"], flush=True)


if __name__ == "__main__":
    index = Index()
    try:
        run(index, Path("evaluation-full.json"))
    finally:
        index.client.close()
