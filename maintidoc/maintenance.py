import asyncio
import hashlib
import hmac
import json
import os
import re
import sqlite3
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import pymupdf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

from .corpus import DATA, EQUIPMENT, MANUAL, ROOT

server = FastMCP("MaintiDoc — maintenance de démonstration", log_level="WARNING")


class HistoryResult(TypedDict):
    equipment_id: str
    history: list[dict[str, str]]


class OpenTicketsResult(TypedDict):
    equipment_id: str
    tickets: list[dict[str, str]]


class TicketResult(TypedDict):
    ticket_id: str
    created_at: str
    already_created: bool


def check_equipment(equipment_id):
    if equipment_id != EQUIPMENT:
        raise ValueError("Équipement inconnu dans le corpus de démonstration.")


def connect():
    root = Path(os.environ.get("MAINTIDOC_DATA", str(DATA)))
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "maintenance.sqlite3", timeout=10)
    db.row_factory = sqlite3.Row
    with db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                request_id TEXT PRIMARY KEY, equipment_id TEXT NOT NULL,
                title TEXT NOT NULL, description TEXT NOT NULL, actor TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return db


@server.tool()
def get_equipment_history(equipment_id: str) -> HistoryResult:
    """Read the four fictional interventions documented on page 7."""
    check_equipment(equipment_id)
    with pymupdf.open(MANUAL) as doc:
        rows = []
        for block in doc[6].get_text("blocks", sort=True):
            if block[4].startswith("INC-"):
                incident, date, asset, observation = block[4].strip().splitlines()
                rows.append(dict(incident=incident, date=date, asset=asset, observation=observation))
    return {"equipment_id": equipment_id, "history": rows}


@server.tool()
def get_open_tickets(equipment_id: str) -> OpenTicketsResult:
    """Read approved tickets in the local demo store."""
    check_equipment(equipment_id)
    db = connect()
    try:
        rows = db.execute("""SELECT request_id AS ticket_id, title, description, created_at
                            FROM tickets WHERE equipment_id = ?
                            ORDER BY created_at DESC, rowid DESC LIMIT 20""", (equipment_id,))
        return {"equipment_id": equipment_id, "tickets": [dict(row) for row in rows]}
    finally:
        db.close()


def sign(secret, actor, request_id, expiry, equipment_id, title, description):
    payload = json.dumps([actor, request_id, expiry, equipment_id, title, description],
                         ensure_ascii=False, separators=(",", ":"))
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


@server.tool()
def create_ticket(equipment_id: str, title: str, description: str, approval_token: str) -> TicketResult:
    """Create one ticket from an approval for the exact content."""
    check_equipment(equipment_id)
    secret = os.environ.get("MAINTIDOC_APPROVAL_SECRET", "")
    actor = os.environ.get("MAINTIDOC_ACTOR", "")
    match = re.fullmatch(r"([0-9a-f]{32})\.(\d{1,12})\.([0-9a-f]{64})", approval_token)
    if not (secret and actor and match and 1 <= len(title.strip()) <= 150
            and 1 <= len(description.strip()) <= 3500):
        raise ValueError("Approbation requise ou contenu invalide.")
    request_id, expiry, digest = match.groups()
    expected = sign(secret, actor, request_id, expiry, equipment_id, title, description)
    if int(expiry) < time.time() or not hmac.compare_digest(digest, expected):
        raise ValueError("Approbation invalide, expirée ou différente du brouillon.")
    db = connect()
    try:
        with db:
            inserted = db.execute("""INSERT INTO tickets (request_id, equipment_id, title, description, actor)
                                     VALUES (?, ?, ?, ?, ?) ON CONFLICT(request_id) DO NOTHING""",
                                  (request_id, equipment_id, title, description, actor))
            record = db.execute("SELECT * FROM tickets WHERE request_id = ?", (request_id,)).fetchone()
            if tuple(record[field] for field in ("equipment_id", "title", "description", "actor")) != (
                    equipment_id, title, description, actor):
                raise ValueError("Cet identifiant est déjà lié à un autre ticket.")
            return {"ticket_id": request_id, "created_at": record["created_at"],
                    "already_created": inserted.rowcount == 0}
    finally:
        db.close()


class Maintenance:
    def __init__(self, root=DATA):
        self.root = root
        self.secret = uuid4().hex + uuid4().hex

    async def _call(self, actor, calls):
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "maintidoc.maintenance"], cwd=str(ROOT),
            env={"MAINTIDOC_DATA": str(self.root), "MAINTIDOC_ACTOR": actor,
                 "MAINTIDOC_APPROVAL_SECRET": self.secret},
        )
        async with asyncio.timeout(20):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=8)) as session:
                    await session.initialize()
                    outputs = {}
                    for tool, args in calls:
                        result = await session.call_tool(tool, args)
                        if result.isError or result.structuredContent is None:
                            detail = getattr(result.content[0], "text", "") if result.content else ""
                            raise RuntimeError(detail or f"Résultat MCP invalide ({tool}).")
                        outputs[tool] = result.structuredContent
                    return outputs

    def read(self, equipment_id, actor, tools):
        return asyncio.run(self._call(actor, [(tool, {"equipment_id": equipment_id}) for tool in tools]))

    def create(self, actor, request_id, draft):
        expiry = str(int(time.time()) + 120)
        digest = sign(self.secret, actor, request_id, expiry,
                      draft["equipment_id"], draft["title"], draft["description"])
        args = {**draft, "approval_token": f"{request_id}.{expiry}.{digest}"}
        return asyncio.run(self._call(actor, [("create_ticket", args)]))["create_ticket"]


if __name__ == "__main__":
    server.run(transport="stdio")
