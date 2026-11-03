from __future__ import annotations

from pathlib import Path
import re
import textwrap

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "conference_paper_ieee_phase2_draft.md"
OUTPUT = ROOT / "output" / "pdf" / "ai_driven_mas_ieee_phase2_paper.pdf"

PAGE_W, PAGE_H = letter
MARGIN_X = 42
MARGIN_TOP = 44
MARGIN_BOTTOM = 42
GUTTER = 16
COL_W = (PAGE_W - (2 * MARGIN_X) - GUTTER) / 2

FONT = "Times-Roman"
BOLD = "Times-Bold"
ITALIC = "Times-Italic"


def clean_text(text: str) -> str:
    replacements = {
        "->": "->",
        "≥": ">=",
        "²": "^2",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text


def wrap_line(text: str, font: str, size: float, width: float) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return [""]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class TwoColumnWriter:
    def __init__(self, c: canvas.Canvas, first_page_top: float | None = None) -> None:
        self.c = c
        self.page = 1
        self.col = 0
        self.first_page_top = first_page_top or PAGE_H - MARGIN_TOP
        self.y = self.column_top()

    def column_top(self) -> float:
        if self.page == 1:
            return self.first_page_top
        return PAGE_H - MARGIN_TOP

    def new_page(self) -> None:
        self._footer()
        self.c.showPage()
        self.page += 1
        self.col = 0
        self.y = self.column_top()

    def next_col(self) -> None:
        if self.col == 0:
            self.col = 1
            self.y = self.column_top()
        else:
            self.new_page()

    def x(self) -> float:
        return MARGIN_X + (COL_W + GUTTER) * self.col

    def ensure(self, height: float) -> None:
        if self.y - height < MARGIN_BOTTOM:
            self.next_col()

    def draw_paragraph(self, text: str, font: str = FONT, size: float = 9.1, leading: float = 10.9, gap: float = 5) -> None:
        lines = wrap_line(text, font, size, COL_W)
        self.ensure(len(lines) * leading + gap)
        self.c.setFont(font, size)
        self.c.setFillColor(colors.black)
        for line in lines:
            self.c.drawString(self.x(), self.y, line)
            self.y -= leading
        self.y -= gap

    def draw_heading(self, text: str) -> None:
        self.ensure(22)
        self.y -= 3
        self.c.setFont(BOLD, 10.0)
        self.c.drawCentredString(self.x() + COL_W / 2, self.y, text.upper())
        self.y -= 14

    def draw_subheading(self, text: str) -> None:
        self.ensure(18)
        self.c.setFont(BOLD, 9.2)
        self.c.drawString(self.x(), self.y, text)
        self.y -= 11

    def draw_bullet(self, text: str) -> None:
        lines = wrap_line(text, FONT, 8.9, COL_W - 12)
        self.ensure(len(lines) * 10.6 + 3)
        self.c.setFont(FONT, 8.9)
        self.c.drawString(self.x(), self.y, "-")
        for idx, line in enumerate(lines):
            self.c.drawString(self.x() + 10, self.y, line)
            self.y -= 10.6
        self.y -= 3

    def draw_table(self, rows: list[list[str]]) -> None:
        if not rows:
            return
        row_h = 18
        self.ensure(row_h * len(rows) + 6)
        left_w = COL_W * 0.38
        right_w = COL_W - left_w
        for idx, row in enumerate(rows):
            y0 = self.y - row_h + 4
            fill = colors.HexColor("#f0f3f6") if idx == 0 else colors.white
            self.c.setFillColor(fill)
            self.c.rect(self.x(), y0, COL_W, row_h, fill=True, stroke=False)
            self.c.setStrokeColor(colors.HexColor("#c9d2dc"))
            self.c.rect(self.x(), y0, COL_W, row_h, fill=False, stroke=True)
            self.c.line(self.x() + left_w, y0, self.x() + left_w, y0 + row_h)
            self.c.setFillColor(colors.black)
            self.c.setFont(BOLD if idx == 0 else FONT, 7.1)
            self.c.drawString(self.x() + 3, y0 + 6, row[0][:28])
            self.c.drawString(self.x() + left_w + 3, y0 + 6, row[1][:54])
            self.y -= row_h
        self.y -= 6

    def _footer(self) -> None:
        self.c.setFont(FONT, 7)
        self.c.setFillColor(colors.HexColor("#555555"))
        self.c.drawCentredString(PAGE_W / 2, 24, f"Page {self.page}")

    def finish(self) -> None:
        self._footer()


def parse_markdown(md: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    lines = clean_text(md).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("**") and line.endswith("**"):
            blocks.append(("boldpara", line.strip("*")))
            i += 1
            continue
        if line.startswith("## "):
            blocks.append(("heading", line[3:].strip()))
            i += 1
            continue
        if line.startswith("### "):
            blocks.append(("subheading", line[4:].strip()))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [part.strip() for part in lines[i].strip().strip("|").split("|")]
                if len(parts) >= 2 and not all(set(part) <= {"-", ":"} for part in parts):
                    rows.append(parts[:2])
                i += 1
            blocks.append(("table", rows))
            continue
        if line.startswith("- "):
            blocks.append(("bullet", line[2:].strip()))
            i += 1
            continue
        if re.match(r"^\d+\. ", line):
            blocks.append(("bullet", re.sub(r"^\d+\. ", "", line)))
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("## ", "### ", "|", "- ")) or re.match(r"^\d+\. ", nxt):
                break
            para.append(nxt)
            i += 1
        blocks.append(("para", " ".join(para)))
    return blocks


def draw_title_page(c: canvas.Canvas, md: str) -> float:
    c.setTitle("AI-Driven Multi-Agent System for Automated End-to-End Data Analytics")
    y = PAGE_H - 48
    title = "AI-Driven Multi-Agent System for Automated End-to-End Data Analytics"
    for line in textwrap.wrap(title, width=58):
        c.setFont(BOLD, 15)
        c.drawCentredString(PAGE_W / 2, y, line)
        y -= 18
    y -= 4
    c.setFont(FONT, 9)
    authors = "A. M. Chiranthan, Sushanth S. Gowda, Vindya Gangotri G. K., Prajwal M. P."
    c.drawCentredString(PAGE_W / 2, y, authors)
    y -= 12
    c.drawCentredString(PAGE_W / 2, y, "Department of Computer Science and Engineering (Data Science)")
    y -= 11
    c.drawCentredString(PAGE_W / 2, y, "ATME College of Engineering, Mysuru, India")
    y -= 20

    abstract_match = re.search(r"## Abstract\s+(.*?)\s+\*\*Keywords:\*\*(.*?)(?:\n\n##|\Z)", md, re.DOTALL)
    if abstract_match:
        abstract = clean_text(abstract_match.group(1).strip())
        keywords = clean_text(abstract_match.group(2).strip())
        c.setFont(BOLD, 9)
        c.drawString(MARGIN_X, y, "Abstract-")
        c.setFont(FONT, 8.8)
        abstract_lines = wrap_line(abstract, FONT, 8.8, PAGE_W - 2 * MARGIN_X - 48)
        x0 = MARGIN_X + 48
        first = True
        for line in abstract_lines:
            c.drawString(x0 if first else MARGIN_X, y, line)
            first = False
            y -= 10.5
        y -= 5
        c.setFont(BOLD, 8.8)
        c.drawString(MARGIN_X, y, "Keywords-")
        c.setFont(ITALIC, 8.8)
        for idx, line in enumerate(wrap_line(keywords, ITALIC, 8.8, PAGE_W - 2 * MARGIN_X - 52)):
            c.drawString(MARGIN_X + 52 if idx == 0 else MARGIN_X, y, line)
            y -= 10.5
    return y - 12


def build_pdf() -> None:
    md = SOURCE.read_text(encoding="utf-8")
    body_md = re.sub(r"^# .*?\n", "", md, count=1, flags=re.DOTALL)
    body_md = re.sub(r"^\*\*A\. M\. Chiranthan.*?\*\*\s*\n", "", body_md, count=1, flags=re.MULTILINE)
    body_md = re.sub(r"^Department of Computer Science.*?\n", "", body_md, count=1, flags=re.MULTILINE)
    body_md = re.sub(r"^ATME College of Engineering.*?\n", "", body_md, count=1, flags=re.MULTILINE)
    body_md = re.sub(r"## Abstract.*?\n\n## I\. Introduction", "## I. Introduction", body_md, flags=re.DOTALL)
    blocks = parse_markdown(body_md)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=letter)
    start_y = draw_title_page(c, md)
    writer = TwoColumnWriter(c, first_page_top=start_y)

    for kind, payload in blocks:
        if kind == "heading":
            writer.draw_heading(str(payload))
        elif kind == "subheading":
            writer.draw_subheading(str(payload))
        elif kind == "bullet":
            writer.draw_bullet(str(payload))
        elif kind == "boldpara":
            writer.draw_paragraph(str(payload), font=BOLD, size=8.3)
        elif kind == "table":
            writer.draw_table(payload)  # type: ignore[arg-type]
        else:
            writer.draw_paragraph(str(payload))

    writer.finish()
    c.save()

    reader = PdfReader(str(OUTPUT))
    if len(reader.pages) > 6:
        raise RuntimeError(f"Generated PDF has {len(reader.pages)} pages; expected 6 or fewer.")
    print(f"Generated {OUTPUT}")
    print(f"Pages: {len(reader.pages)}")


if __name__ == "__main__":
    build_pdf()
