"""Private, read-only Telegram client. Use a Qdrant server for concurrent web access."""
import os
from uuid import uuid4

import httpx

from .corpus import EQUIPMENT
from .maintenance import Maintenance
from .retrieval import Index
from .workflow import build_workflow


def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    allowed = set(os.environ["TELEGRAM_CHAT_IDS"].split(","))
    index = Index()
    graph = build_workflow(index, Maintenance())
    offset = 0
    try:
        with httpx.Client(base_url=f"https://api.telegram.org/bot{token}/", timeout=40) as api:
            while True:
                response = api.post("getUpdates", json={"offset": offset, "timeout": 25})
                response.raise_for_status()
                for update in response.json()["result"]:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    chat = msg.get("chat", {})
                    if chat.get("type") != "private" or str(chat["id"]) not in allowed:
                        continue
                    question = msg.get("text", "").strip()[:1200]
                    if not question:
                        continue
                    thread = uuid4().hex
                    try:
                        state = graph.invoke({"question":question, "role":"reader", "actor":f"telegram-{chat['id']}/{thread}",
                                              "equipment_id":EQUIPMENT}, {"configurable":{"thread_id":thread}})
                        answer = state["answer"]
                        lines = [f'{c["statement"]} (p. {state["evidence"][c["source"]-1]["page"]})'
                                 for c in answer["claims"]]
                        text = "\n".join([*lines, answer["clarification"]]).strip()
                        api.post("sendMessage", json={"chat_id":chat["id"], "text":text[:4000]}).raise_for_status()
                    finally:
                        graph.checkpointer.delete_thread(thread)
    finally:
        index.client.close()


if __name__ == "__main__":
    main()
