import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
from docx import Document
from PIL import Image
from PyPDF2 import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

try:
    import pytesseract
except Exception:
    pytesseract = None


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
HISTORY_FILE = BASE_DIR / "history.json"
EXPORTS_DIR = BASE_DIR / "exports"


def ensure_directories() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    EXPORTS_DIR.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def read_text_from_file(file_path: str) -> Tuple[str, str]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore"), "TXT"
    if suffix == ".docx":
        document = Document(path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs), "DOCX"
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages), "PDF"
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}:
        image = Image.open(path)
        if pytesseract is None:
            return "", "IMAGE"
        return pytesseract.image_to_string(image), "IMAGE"

    raise ValueError("Unsupported file type. Please upload TXT, DOCX, PDF, or image files.")


def save_history(entry: Dict[str, Any]) -> None:
    ensure_directories()
    current = load_history()
    current.insert(0, entry)
    HISTORY_FILE.write_text(json.dumps(current[:50], indent=2), encoding="utf-8")


def load_history() -> List[Dict[str, Any]]:
    ensure_directories()
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def create_probability_figure(ai_probability: float, human_probability: float, dark_mode: bool) -> Figure:
    facecolor = "#1f1f1f" if dark_mode else "#ffffff"
    text_color = "#f5f5f5" if dark_mode else "#202124"
    figure = Figure(figsize=(4.0, 3.2), dpi=100, facecolor=facecolor)
    ax = figure.add_subplot(111)
    ax.set_facecolor(facecolor)
    colors_list = ["#ff6b6b", "#4dd0a8"]
    ax.pie(
        [ai_probability, human_probability],
        labels=["AI", "Human"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors_list,
        textprops={"color": text_color, "fontsize": 10},
    )
    ax.set_title("Probability Split", color=text_color, fontsize=12)
    return figure


def create_metrics_figure(metrics: Dict[str, float], dark_mode: bool) -> Figure:
    facecolor = "#1f1f1f" if dark_mode else "#ffffff"
    text_color = "#f5f5f5" if dark_mode else "#202124"
    figure = Figure(figsize=(5.2, 3.2), dpi=100, facecolor=facecolor)
    ax = figure.add_subplot(111)
    ax.set_facecolor(facecolor)
    labels = list(metrics.keys())
    values = list(metrics.values())
    bars = ax.bar(labels, values, color=["#6c8cff", "#ffb86b", "#f78fb3", "#82ccdd", "#78e08f"])
    ax.tick_params(axis="x", rotation=20, labelcolor=text_color)
    ax.tick_params(axis="y", labelcolor=text_color)
    ax.set_title("Metric Scores", color=text_color, fontsize=12)
    ax.spines["bottom"].set_color(text_color)
    ax.spines["left"].set_color(text_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            color=text_color,
            fontsize=8,
        )
    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    canvas.print_png(buffer)
    return buffer.getvalue()


def export_result_txt(report_text: str, destination: str) -> None:
    Path(destination).write_text(report_text, encoding="utf-8")


def export_report_pdf(result: Dict[str, Any], destination: str) -> None:
    document = SimpleDocTemplate(destination, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AI Content Detector Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 12),
    ]

    table_data = [
        ["AI Probability", f"{result['ai_probability']:.2f}%"],
        ["Human Probability", f"{result['human_probability']:.2f}%"],
        ["Verdict", result["verdict"]],
        ["Source", result.get("source_label", "Manual Input")],
    ]
    table = Table(table_data, colWidths=[180, 240])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6e9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([table, Spacer(1, 14)])

    story.append(Paragraph(result["summary"], styles["BodyText"]))
    story.append(Spacer(1, 12))

    metrics_rows = [["Metric", "Value"]]
    metrics_rows.extend([[name, f"{value:.4f}"] for name, value in result["metrics"].items()])
    metrics_table = Table(metrics_rows, colWidths=[180, 240])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#74b9ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(metrics_table)
    document.build(story)


def format_result_report(result: Dict[str, Any]) -> str:
    lines = [
        "AI Content Detector Analysis Report",
        "=" * 34,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source: {result.get('source_label', 'Manual Input')}",
        f"AI Probability: {result['ai_probability']:.2f}%",
        f"Human Probability: {result['human_probability']:.2f}%",
        f"Final Verdict: {result['verdict']}",
        "",
        "Metrics:",
    ]
    lines.extend(f"- {name}: {value:.4f}" for name, value in result["metrics"].items())
    lines.extend(["", f"Summary: {result['summary']}"])
    return "\n".join(lines)
