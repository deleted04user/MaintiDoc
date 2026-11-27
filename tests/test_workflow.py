import asyncio
import tempfile
import unittest
from pathlib import Path

from langgraph.types import Command

from maintidoc.corpus import EQUIPMENT, MANUAL
from maintidoc.maintenance import Maintenance
from maintidoc.retrieval import Index, lexical
from maintidoc.workflow import Answer, Claim, build_workflow


class TestEncoders:
    signature = "test-only"

    def texts(self, values):
        return [[1.0, 0.0] for _ in values]

    images = texts

    def rank(self, query, values):
        words = set(lexical(query).indices)
        return [len(words.intersection(lexical(value).indices)) for value in values]


class MaintiDocTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.storage.cleanup)
        cls.root = Path(cls.storage.name)
        index = Index(cls.root, TestEncoders())
        try:
            cls.manifest = index.index(MANUAL)
            assert index.index(MANUAL) == cls.manifest
        finally:
            index.client.close()
        cls.index = Index(cls.root, TestEncoders())
        cls.addClassCleanup(cls.index.client.close)
        cls.maintenance = Maintenance(cls.root)

    def test_retrieves_persisted_alarm_and_figure(self):
        alarm = self.index.search("Que signifie AL-01 ?")[0]
        self.assertEqual((alarm["element_id"], alarm["page"], alarm["revision"]), ("AL-01", 5, "1.1"))
        self.assertIn("Chemin FC degrade", alarm["table"].values())
        photo = (self.root / "assets" / self.manifest["generation"] / "FIG-02.png").read_bytes()
        figure = self.index.search("Schéma FIG-02", photo)[0]
        self.assertEqual((figure["element_id"], figure["page"]), ("FIG-02", 3))
        self.assertIn("OCR:", figure["text"])

    def test_retries_invalid_citation(self):
        calls = []

        def generate(prompt, schema, payload, images=()):
            calls.append(payload)
            quote = "Citation inventée pour le test." if len(calls) == 1 else "Chemin FC degrade"
            return Answer(claims=[Claim(source=1, statement=quote, quote=quote)])

        graph = build_workflow(self.index, self.maintenance, generate)
        result = graph.invoke(self.state("Que signifie AL-01 ?"), self.config("citations"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["answer"]["claims"][0]["quote"], "Chemin FC degrade")

    def test_unsourced_model_clarification_cannot_override_alarm_reference(self):
        def generate(prompt, schema, payload, images=()):
            return Answer(clarification="Le voyant impose un remplacement immédiat.")

        graph = build_workflow(self.index, self.maintenance, generate)
        result = graph.invoke(self.state("Que signifie AL-01 ?"), self.config("fallback"))
        self.assertEqual(result["answer"]["clarification"], "")
        self.assertIn("Chemin FC degrade", result["answer"]["claims"][0]["quote"])

    def test_mcp_requires_approval_and_creates_exact_draft_once(self):
        # Extractive model stand-in. The MCP transport and ticket store are real.
        def generate(prompt, schema, payload, images=()):
            quote = "Chemin FC degrade"
            return Answer(claims=[Claim(source=1, statement=quote, quote=quote)])

        graph = build_workflow(self.index, self.maintenance, generate)
        state = self.state("Consulte l’historique et prépare un ticket pour AL-01.")
        result = graph.invoke(state, self.config("refused"))
        self.assertTrue(result["__interrupt__"])
        tools = result["tool_result"]
        self.assertEqual(len(tools["get_equipment_history"]["history"]), 4)
        self.assertEqual(tools["get_open_tickets"]["tickets"], [])
        denied = graph.invoke(Command(resume=False), self.config("refused"))
        self.assertNotIn("ticket", denied)
        pending = graph.invoke(state, self.config("approved"))
        draft = pending["draft"]
        with self.assertRaises(ExceptionGroup):
            asyncio.run(self.maintenance._call("technician", [
                ("create_ticket", {**draft, "approval_token": "invalid"}),
            ]))
        approved = graph.invoke(Command(resume=True), self.config("approved"))
        self.assertEqual(approved["ticket"]["ticket_id"], pending["request_id"])
        retry = self.maintenance.create("technician", pending["request_id"], draft)
        self.assertTrue(retry["already_created"])
        data = self.maintenance.read(EQUIPMENT, "technician", ["get_open_tickets"])
        tickets = data["get_open_tickets"]["tickets"]
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0]["description"], draft["description"])

    def test_orange_light_prompts_for_clarification(self):
        def unused_model(*args, **kwargs):
            self.fail("The model should never be called for this ambiguous alert.")

        graph = build_workflow(self.index, self.maintenance, unused_model)
        result = graph.invoke(self.state("Quelle est la cause du voyant orange ?"), self.config("orange"))
        self.assertIn("code d’alarme", result["answer"]["clarification"])

    @staticmethod
    def state(question):
        return {"question": question, "actor": "technician", "equipment_id": EQUIPMENT}

    @staticmethod
    def config(thread):
        return {"configurable": {"thread_id": thread}, "recursion_limit": 14}
