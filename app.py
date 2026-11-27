from io import BytesIO
import logging
import time
from uuid import uuid4

import streamlit as st
from streamlit.logger import get_logger
from langgraph.types import Command
from PIL import Image, ImageOps

from maintidoc.corpus import DATA, DOCUMENT, EQUIPMENT
from maintidoc.access import identify
from maintidoc.maintenance import Maintenance
from maintidoc.retrieval import Index
from maintidoc.workflow import build_workflow

st.set_page_config(page_title="MaintiDoc", layout="wide")
st.title("MaintiDoc")
st.caption("Assistant documentaire de maintenance • SAN-LAB 2000 • Banc d’essai fictif")


@st.cache_resource
def resources(_index_factory):
    return _index_factory(), Maintenance()


def normalize_photo(upload):
    with Image.open(upload) as image:
        if image.width * image.height > 24_000_000:
            raise ValueError("La photo dépasse 24 mégapixels.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1280, 1280))
        output = BytesIO()
        image.save(output, format="JPEG", quality=88)
        return output.getvalue()


account, role = identify()
if not (DATA / "active-index.json").exists():
    st.info("Le manuel doit être indexé avant la première recherche.")
    st.code("python -m maintidoc.retrieval", language="powershell")
    st.stop()

index, maintenance = resources(globals().get("index_factory", Index))
session = st.session_state
if "actor" not in session:
    session.actor = f"{account}/{uuid4().hex}"
    session.graph = build_workflow(index, maintenance)
    session.config = None
    session.result = {}
    session.pending = False

main, sources = st.columns([3, 2], gap="large")
with main:
    st.subheader("Diagnostic documentaire")
    with st.form("question"):
        question = st.text_area("Question sur le SAN", placeholder="Que signifie l’alarme AL-01 ?",
                                max_chars=1200, disabled=session.pending)
        upload = st.file_uploader("Photo du panneau ou du composant", type=["png", "jpg", "jpeg", "webp"],
                                  disabled=session.pending)
        prepare = st.checkbox("Préparer un ticket", disabled=session.pending or role != "technician")
        submit = st.form_submit_button("Interroger le manuel", type="primary", disabled=session.pending)
    if submit:
        if not question.strip():
            st.warning("Saisissez une question.")
        else:
            try:
                photo = normalize_photo(upload) if upload else None
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                st.error(f"Photo illisible. {error}")
            else:
                config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 14}
                try:
                    started = time.monotonic()
                    with st.spinner("Recherche dans le manuel…"):
                        result = session.graph.invoke({
                            "question": question.strip(), "photo": photo, "actor": session.actor,
                            "equipment_id": EQUIPMENT, "ticket_requested": prepare, "role": role,
                        }, config)
                except (OSError, ValueError, RuntimeError, ImportError) as error:
                    st.error(f"Recherche indisponible. {error}")
                    session.graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
                else:
                    get_logger("maintidoc.audit").log(
                        logging.INFO, "search role=%s elapsed_ms=%d sources=%d draft=%s",
                        role, (time.monotonic()-started)*1000, len(result.get("evidence", [])),
                        bool(result.get("draft")))
                    if session.config:
                        session.graph.checkpointer.delete_thread(
                            session.config["configurable"]["thread_id"])
                    session.result = result
                    session.config = config
                    session.pending = bool(result.get("__interrupt__"))
                    st.rerun()

    result = session.result
    if result:
        st.divider()
        st.subheader("Réponse")
        if result.get("response_note"):
            st.info(result["response_note"])
        answer = result.get("answer", {})
        for claim in answer.get("claims", []):
            st.text(f'[{claim["source"]}] {claim["statement"]}')
        if answer.get("clarification"):
            st.warning(answer["clarification"])
        if result.get("tool_error"):
            st.warning(result["tool_error"])

        mcp = result.get("tool_result", {})
        if "get_equipment_history" in mcp:
            with st.expander("Historique des interventions", expanded=True):
                for item in mcp["get_equipment_history"]["history"]:
                    st.text(f'{item["incident"]} · {item["date"]} · {item["asset"]}')
                    st.caption(item["observation"])
        if "get_open_tickets" in mcp:
            with st.expander("Tickets ouverts", expanded=True):
                tickets = mcp["get_open_tickets"]["tickets"]
                if tickets:
                    for ticket in tickets:
                        st.text(f'{ticket["title"]} · {ticket["ticket_id"][:8]}')
                else:
                    st.caption("Aucun ticket ouvert.")

        if session.pending:
            draft = result["draft"]
            st.subheader("Ticket à valider")
            st.caption(draft["equipment_id"])
            st.text(draft["title"])
            st.text(draft["description"])
            retry = session.graph.get_state(session.config).next == ("create",)
            approve, reject = st.columns(2)
            accepted = approve.button("Réessayer la création" if retry else "Approuver et créer",
                                      type="primary")
            declined = reject.button("Refuser le brouillon", disabled=retry)
            if accepted or declined:
                try:
                    session.result = session.graph.invoke(None if retry else Command(resume=accepted),
                                                          session.config)
                except (RuntimeError, OSError, TimeoutError, ExceptionGroup) as error:
                    st.error(f"Création indisponible. {error}")
                else:
                    session.pending = False
                    st.rerun()
        elif result.get("ticket"):
            st.success(f'Ticket créé · {result["ticket"]["ticket_id"]}')
        elif result.get("approved") is False:
            st.info("Brouillon refusé.")

with sources:
    st.subheader("Sources")
    evidence = session.result.get("evidence", [])
    cited = {claim["source"] for claim in session.result.get("answer", {}).get("claims", [])}
    if not evidence:
        st.caption(f"Les extraits du manuel {DOCUMENT} apparaîtront ici.")
    for number, row in enumerate(evidence, 1):
        with st.expander(f'[{number}] {row["element_id"]} · p. {row["page"]}',
                         expanded=number in cited or (not cited and number == 1)):
            st.caption(f'{row["document_id"]} · révision {row["revision"]}')
            if row.get("image"):
                st.image(str(index.image_path(row)))
            st.text(row["text"])
            for claim in session.result.get("answer", {}).get("claims", []):
                if claim["source"] == number and claim["quote"] != claim["statement"]:
                    st.caption(f'Citation vérifiée — « {claim["quote"]} »')
    if evidence:
        path = DATA / "assets" / evidence[0]["generation"] / "source.pdf"
        st.download_button("Télécharger le manuel source", path.read_bytes(),
                           file_name=evidence[0]["filename"], mime="application/pdf")
