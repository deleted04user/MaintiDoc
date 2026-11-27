import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maintidoc.corpus import EQUIPMENT
from maintidoc.maintenance import Maintenance


class AccessTest(unittest.TestCase):
    def test_role_checked_again_after_draft_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accounts = root / "accounts.json"
            accounts.write_text('{"alice":{"role":"reader"}}')
            client = Maintenance(root)
            draft = {"title":"Vérifier", "description":"Alarme confirmée", "equipment_id":EQUIPMENT}
            with patch.dict(os.environ, {"MAINTIDOC_ACCOUNTS":str(accounts)}):
                with self.assertRaises(ExceptionGroup):
                    client.create("alice/session", "1"*32, draft)
                with self.assertRaises(ValueError):
                    client.read(EQUIPMENT, "unknown/session", ["get_equipment_history"])
                result = client.read(EQUIPMENT, "alice/session", ["get_open_tickets"])
                self.assertEqual(result["get_open_tickets"]["tickets"], [])
