import os
import re
import subprocess
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("MAINTIDOC_DATA", ROOT / ".maintidoc")).resolve()
MANUAL = ROOT / "data" / "Manuel_Maintenance_SAN_Projet_Integrateur.pdf"
DOCUMENT = "SAN-MAN-001"
EQUIPMENT = "SAN-LAB-2000"

TABLES = {
    2: [(142, 307, "PERIMETRE")],
    3: [(337, 478, "LEXIQUE-FC")],
    4: [(351, 467, "COMPOSANTS")],
    5: [(351, 492, "ALARM-TAB-01")],
    6: [(505, 621, "AUTORISATIONS")],
    7: [(93, 209, "HISTORIQUE"), (257, 398, "EXEMPLE-TICKET")],
}
PROCEDURES = {3: (503, 581, "DIAG-FC-01"), 4: (492, 549, "DISC-02"),
              6: (349, 422, "PROC-SAN-01")}


def read_ocr(image: bytes, mode: str = "6") -> str:
    try:
        result = subprocess.run(
            [os.getenv("TESSERACT", "tesseract"), "stdin", "stdout", "-l",
             os.getenv("OCR_LANG", "fra+eng"), "--psm", mode],
            input=image, capture_output=True, check=True, timeout=25,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract a refusé l’image. {detail}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Le délai de reconnaissance du texte de l’image est dépassé.") from error
    return result.stdout.decode("utf-8").strip()


def tidy(text: str) -> str:
    return " ".join(text.split())


def extract(path: Path, assets: Path, ocr=read_ocr) -> list[dict]:
    assets.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(path) as doc:
        cover = doc[0].get_text()
        if not cover.strip():
            cover = ocr(doc[0].get_pixmap(dpi=180).tobytes("png"))
        if DOCUMENT not in cover:
            raise ValueError(f"Le corpus attend le manuel {DOCUMENT}.")
        revision = re.search(r"Revision\s+([\d.]+)", cover)
        rows = []
        for page in doc:
            number = page.number + 1
            if number >= 8:
                # The supplied manual ends with a separator and a reference index.
                continue
            blocks = [b for b in page.get_text("blocks", sort=True)
                      if b[6] == 0 and b[1] < 800]
            heading = tidy(blocks[0][4]) if blocks else f"Page {number}"

            def add(element, kind, text, **extra):
                rows.append(dict(document_id=DOCUMENT, revision=revision[1] if revision else "",
                                 equipment_id=EQUIPMENT, filename=path.name, page=number,
                                 element_id=element, kind=kind, heading=heading,
                                 text=tidy(text), **extra))

            if not blocks:
                pixmap = page.get_pixmap(dpi=180)
                text = ocr(pixmap.tobytes("png"))
                if text:
                    add(f"OCR-P{number}", "ocr", text)
                continue

            excluded = []
            for top, bottom, table in TABLES.get(number, []):
                table_rows = [b[4].strip().splitlines() for b in blocks if top <= b[1] < bottom]
                if not table_rows:
                    continue
                headers, *values = table_rows
                for index, cells in enumerate(values, 1):
                    if len(cells) != len(headers):
                        raise ValueError(f"Colonnes inattendues dans {table}, page {number}.")
                    record = dict(zip(headers, cells))
                    element = cells[0] if re.fullmatch(r"(AL|INC)-\d+", cells[0]) else f"{table}-{index}"
                    add(element, "table", f"{table} | " + " | ".join(
                        f"{key}: {value}" for key, value in record.items()), table=record)
                excluded.append((top, bottom))

            if number in PROCEDURES:
                top, bottom, element = PROCEDURES[number]
                add(element, "procedure", " ".join(b[4] for b in blocks if top <= b[1] < bottom))
                excluded.append((top, bottom))

            for image in page.get_image_info():
                if not 2 <= number <= 6:
                    continue
                rect = pymupdf.Rect(image["bbox"])
                captions = [b for b in blocks if rect.y1 <= b[1] <= rect.y1 + 30]
                caption = tidy(" ".join(b[4] for b in captions))
                match = re.match(r"FIG-\d+", caption)
                if not match:
                    continue
                filename = f"{match[0]}.png"
                raw = page.get_pixmap(clip=rect, dpi=160).tobytes("png")
                (assets / filename).write_bytes(raw)
                labels = ocr(raw, "11")
                add(match[0], "figure", caption + (f" | OCR: {labels}" if labels else ""),
                    image=filename)
                excluded.extend((b[1], b[3]) for b in captions)

            # The example questions are acceptance data rather than retrieval evidence.
            if number == 7:
                excluded.append((425, 800))
            paragraphs = []
            end = 0
            for block in blocks:
                if any(top <= block[1] <= bottom for top, bottom in excluded):
                    continue
                if not paragraphs or block[1] - end > 8:
                    paragraphs.append("")
                paragraphs[-1] += " " + block[4]
                end = block[3]
            for index, paragraph in enumerate(paragraphs, 1):
                add(f"P{number}-{index}", "text", paragraph)
    return rows
