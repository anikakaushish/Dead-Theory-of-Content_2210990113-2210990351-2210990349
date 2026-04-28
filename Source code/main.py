import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from detector import AIContentDetector
from utils import (
    EXPORTS_DIR,
    create_metrics_figure,
    create_probability_figure,
    ensure_directories,
    export_report_pdf,
    export_result_txt,
    format_result_report,
    load_history,
    read_text_from_file,
    save_history,
)


class AIContentDetectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        ensure_directories()
        self.title("AI Content Detector")
        self.geometry("1320x840")
        self.minsize(1160, 760)

        self.detector = AIContentDetector()
        self.dark_mode = True
        self.current_result = None
        self.current_source_label = "Manual Input"
        self.probability_canvas = None
        self.metrics_canvas = None

        self.theme_colors = {}
        self._configure_root()
        self._build_styles()
        self._build_layout()
        self._apply_theme()
        self._load_history_panel()

    def _configure_root(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

    def _build_styles(self) -> None:
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

    def _build_layout(self) -> None:
        self.main_frame = ttk.Frame(self, padding=16)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=2)
        self.main_frame.rowconfigure(3, weight=2)

        self.sidebar = ttk.Frame(self, padding=(0, 16, 16, 16))
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar.rowconfigure(2, weight=1)

        header = ttk.Frame(self.main_frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(header, text="AI Content Detector", style="Title.TLabel")
        self.title_label.grid(row=0, column=0, sticky="w")

        self.subtitle_label = ttk.Label(
            header,
            text="Analyze writing style, perplexity, repetition, and burstiness in one desktop app.",
            style="Subtitle.TLabel",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.mode_var = tk.BooleanVar(value=self.dark_mode)
        self.mode_toggle = ttk.Checkbutton(
            header,
            text="Dark mode",
            variable=self.mode_var,
            command=self.toggle_theme,
            style="Switch.TCheckbutton",
        )
        self.mode_toggle.grid(row=0, column=1, sticky="e")

        input_card = ttk.LabelFrame(self.main_frame, text="Input Text", padding=14)
        input_card.grid(row=1, column=0, sticky="nsew", pady=(0, 16))
        input_card.columnconfigure(0, weight=1)
        input_card.rowconfigure(0, weight=1)

        self.text_input = tk.Text(
            input_card,
            wrap="word",
            height=14,
            relief="flat",
            bd=0,
            font=("Segoe UI", 11),
            insertwidth=2,
        )
        self.text_input.grid(row=0, column=0, sticky="nsew")

        button_bar = ttk.Frame(input_card)
        button_bar.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for column in range(6):
            button_bar.columnconfigure(column, weight=1)

        self.analyze_button = ttk.Button(button_bar, text="Analyze Text", command=self.start_analysis, style="Accent.TButton")
        self.analyze_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.clear_button = ttk.Button(button_bar, text="Clear", command=self.clear_all)
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=4)

        self.upload_button = ttk.Button(button_bar, text="Upload File", command=self.upload_file)
        self.upload_button.grid(row=0, column=2, sticky="ew", padx=4)

        self.copy_button = ttk.Button(button_bar, text="Copy Result", command=self.copy_result)
        self.copy_button.grid(row=0, column=3, sticky="ew", padx=4)

        self.export_txt_button = ttk.Button(button_bar, text="Export TXT", command=self.export_txt)
        self.export_txt_button.grid(row=0, column=4, sticky="ew", padx=4)

        self.export_pdf_button = ttk.Button(button_bar, text="Save PDF Report", command=self.export_pdf)
        self.export_pdf_button.grid(row=0, column=5, sticky="ew", padx=(8, 0))

        self.progress = ttk.Progressbar(self.main_frame, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        output_card = ttk.LabelFrame(self.main_frame, text="Analysis Output", padding=14)
        output_card.grid(row=3, column=0, sticky="nsew")
        output_card.columnconfigure(0, weight=1)
        output_card.columnconfigure(1, weight=1)
        output_card.rowconfigure(1, weight=1)

        stats_frame = ttk.Frame(output_card)
        stats_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for column in range(3):
            stats_frame.columnconfigure(column, weight=1)

        self.ai_prob_var = tk.StringVar(value="0.00%")
        self.human_prob_var = tk.StringVar(value="0.00%")
        self.verdict_var = tk.StringVar(value="Waiting for analysis")
        self.summary_var = tk.StringVar(value=self.detector.model_status)

        self._create_stat_card(stats_frame, 0, "AI Probability", self.ai_prob_var)
        self._create_stat_card(stats_frame, 1, "Human Probability", self.human_prob_var)
        self._create_stat_card(stats_frame, 2, "Final Verdict", self.verdict_var)

        self.summary_label = ttk.Label(output_card, textvariable=self.summary_var, style="Summary.TLabel", wraplength=740)
        self.summary_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        self.chart_frame = ttk.Frame(output_card)
        self.chart_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.chart_frame.columnconfigure(0, weight=1)
        self.chart_frame.columnconfigure(1, weight=1)
        self.chart_frame.rowconfigure(0, weight=1)

        self.probability_container = ttk.Frame(self.chart_frame)
        self.probability_container.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.metrics_container = ttk.Frame(self.chart_frame)
        self.metrics_container.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        model_frame = ttk.LabelFrame(self.sidebar, text="Detector Status", padding=14)
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        self.model_label = ttk.Label(model_frame, text=self.detector.model_status, wraplength=330, style="Body.TLabel")
        self.model_label.grid(row=0, column=0, sticky="w")

        history_frame = ttk.LabelFrame(self.sidebar, text="Scan History", padding=14)
        history_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 16))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)

        self.history_list = tk.Listbox(history_frame, height=12, relief="flat", activestyle="none")
        self.history_list.grid(row=0, column=0, sticky="nsew")
        self.history_list.bind("<<ListboxSelect>>", self.load_selected_history)

        hints_frame = ttk.LabelFrame(self.sidebar, text="Tips", padding=14)
        hints_frame.grid(row=2, column=0, sticky="nsew")
        hints_frame.columnconfigure(0, weight=1)
        hints_frame.rowconfigure(0, weight=1)

        tips_text = (
            "Upload TXT, DOCX, PDF, or image files.\n\n"
            "For image uploads, OCR is used when Tesseract is installed.\n\n"
            "Longer samples usually produce more stable scores.\n\n"
            "Short question lists may be marked inconclusive when there is not enough context."
        )
        self.tips_label = ttk.Label(hints_frame, text=tips_text, wraplength=320, style="Body.TLabel")
        self.tips_label.grid(row=0, column=0, sticky="nw")

    def _create_stat_card(self, parent: ttk.Frame, column: int, label: str, variable: tk.StringVar) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=0, column=column, sticky="ew", padx=6)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text=label, style="StatLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, textvariable=variable, style="StatValue.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _apply_theme(self) -> None:
        if self.dark_mode:
            self.theme_colors = {
                "bg": "#10141b",
                "panel": "#18202c",
                "card": "#1f2937",
                "text": "#f8fafc",
                "muted": "#aeb8c5",
                "accent": "#6c8cff",
                "success": "#4dd0a8",
                "input": "#0f1722",
            }
        else:
            self.theme_colors = {
                "bg": "#eef2f7",
                "panel": "#ffffff",
                "card": "#f7faff",
                "text": "#16202b",
                "muted": "#4f6071",
                "accent": "#355cff",
                "success": "#0f9d76",
                "input": "#ffffff",
            }

        colors = self.theme_colors
        self.configure(bg=colors["bg"])
        self.style.configure(".", background=colors["bg"], foreground=colors["text"])
        self.style.configure("TFrame", background=colors["bg"])
        self.style.configure("Card.TFrame", background=colors["card"])
        self.style.configure("TLabel", background=colors["bg"], foreground=colors["text"], font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", font=("Segoe UI Semibold", 24), foreground=colors["text"])
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground=colors["muted"])
        self.style.configure("Body.TLabel", foreground=colors["muted"])
        self.style.configure("Summary.TLabel", font=("Segoe UI", 10), foreground=colors["muted"])
        self.style.configure("StatLabel.TLabel", background=colors["card"], foreground=colors["muted"], font=("Segoe UI", 10))
        self.style.configure("StatValue.TLabel", background=colors["card"], foreground=colors["text"], font=("Segoe UI Semibold", 18))
        self.style.configure("TLabelframe", background=colors["panel"], foreground=colors["text"])
        self.style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["text"], font=("Segoe UI Semibold", 10))
        self.style.configure("TButton", padding=8, font=("Segoe UI Semibold", 10))
        self.style.configure("Accent.TButton", background=colors["accent"], foreground="#ffffff")
        self.style.map(
            "Accent.TButton",
            background=[("active", colors["accent"]), ("pressed", colors["accent"])],
            foreground=[("disabled", "#d1d5db"), ("!disabled", "#ffffff")],
        )
        self.style.configure(
            "Horizontal.TProgressbar",
            troughcolor=colors["card"],
            background=colors["accent"],
            bordercolor=colors["card"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
        )
        self.style.configure("Switch.TCheckbutton", background=colors["bg"], foreground=colors["text"])

        self.text_input.configure(
            bg=colors["input"],
            fg=colors["text"],
            insertbackground=colors["text"],
            selectbackground=colors["accent"],
            selectforeground="#ffffff",
        )
        self.history_list.configure(
            bg=colors["input"],
            fg=colors["text"],
            selectbackground=colors["accent"],
            selectforeground="#ffffff",
            highlightthickness=0,
        )

        self._refresh_charts()

    def toggle_theme(self) -> None:
        self.dark_mode = self.mode_var.get()
        self._apply_theme()

    def start_analysis(self) -> None:
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No Text", "Paste or upload some text before running the detector.")
            return

        self.progress.start(12)
        self.analyze_button.configure(state="disabled")
        worker = threading.Thread(target=self._analyze_in_background, args=(text,), daemon=True)
        worker.start()

    def _analyze_in_background(self, text: str) -> None:
        try:
            result = self.detector.analyze_text(text)
            payload = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_label": self.current_source_label,
                "input_preview": text[:180].replace("\n", " "),
                "ai_probability": result.ai_probability,
                "human_probability": result.human_probability,
                "verdict": result.verdict,
                "metrics": result.metrics,
                "summary": result.summary,
            }
            save_history(payload)
            self.after(0, lambda: self._handle_analysis_success(payload))
        except Exception as exc:
            self.after(0, lambda: self._handle_analysis_error(str(exc)))

    def _handle_analysis_success(self, payload) -> None:
        self.progress.stop()
        self.analyze_button.configure(state="normal")
        self.current_result = payload
        self.ai_prob_var.set(f"{payload['ai_probability']:.2f}%")
        self.human_prob_var.set(f"{payload['human_probability']:.2f}%")
        self.verdict_var.set(payload["verdict"])
        self.summary_var.set(payload["summary"])
        self._refresh_charts()
        self._load_history_panel()

    def _handle_analysis_error(self, message: str) -> None:
        self.progress.stop()
        self.analyze_button.configure(state="normal")
        messagebox.showerror("Analysis Error", message)

    def _refresh_charts(self) -> None:
        if self.probability_canvas:
            self.probability_canvas.get_tk_widget().destroy()
        if self.metrics_canvas:
            self.metrics_canvas.get_tk_widget().destroy()

        if not self.current_result:
            return

        probability_figure = create_probability_figure(
            self.current_result["ai_probability"],
            self.current_result["human_probability"],
            self.dark_mode,
        )
        metrics_figure = create_metrics_figure(self.current_result["metrics"], self.dark_mode)

        self.probability_canvas = FigureCanvasTkAgg(probability_figure, master=self.probability_container)
        self.probability_canvas.draw()
        self.probability_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.metrics_canvas = FigureCanvasTkAgg(metrics_figure, master=self.metrics_container)
        self.metrics_canvas.draw()
        self.metrics_canvas.get_tk_widget().pack(fill="both", expand=True)

    def upload_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select a file to analyze",
            filetypes=[
                ("Supported Files", "*.txt *.docx *.pdf *.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("Text Files", "*.txt"),
                ("Word Documents", "*.docx"),
                ("PDF Files", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
            ],
        )
        if not file_path:
            return

        try:
            extracted_text, source_type = read_text_from_file(file_path)
            if not extracted_text.strip():
                if source_type == "IMAGE":
                    raise ValueError("No text could be extracted from the image. Install Tesseract OCR or try another file.")
                raise ValueError("The selected file did not contain readable text.")
            self.text_input.delete("1.0", "end")
            self.text_input.insert("1.0", extracted_text)
            self.current_source_label = f"{source_type}: {Path(file_path).name}"
        except Exception as exc:
            messagebox.showerror("Upload Error", str(exc))

    def clear_all(self) -> None:
        self.text_input.delete("1.0", "end")
        self.ai_prob_var.set("0.00%")
        self.human_prob_var.set("0.00%")
        self.verdict_var.set("Waiting for analysis")
        self.summary_var.set(self.detector.model_status)
        self.current_result = None
        self.current_source_label = "Manual Input"
        self._refresh_charts()

    def copy_result(self) -> None:
        if not self.current_result:
            messagebox.showinfo("Nothing to Copy", "Run an analysis first.")
            return
        report = format_result_report(self.current_result)
        self.clipboard_clear()
        self.clipboard_append(report)
        messagebox.showinfo("Copied", "The analysis report was copied to the clipboard.")

    def export_txt(self) -> None:
        if not self.current_result:
            messagebox.showinfo("No Result", "Run an analysis before exporting.")
            return
        default_name = EXPORTS_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        destination = filedialog.asksaveasfilename(
            title="Export analysis as text",
            initialfile=default_name.name,
            initialdir=EXPORTS_DIR,
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
        )
        if not destination:
            return
        export_result_txt(format_result_report(self.current_result), destination)
        messagebox.showinfo("Export Complete", "The TXT report has been saved.")

    def export_pdf(self) -> None:
        if not self.current_result:
            messagebox.showinfo("No Result", "Run an analysis before saving a PDF report.")
            return
        default_name = EXPORTS_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        destination = filedialog.asksaveasfilename(
            title="Save PDF analysis report",
            initialfile=default_name.name,
            initialdir=EXPORTS_DIR,
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not destination:
            return
        export_report_pdf(self.current_result, destination)
        messagebox.showinfo("Report Saved", "The PDF report has been created.")

    def _load_history_panel(self) -> None:
        self.history_items = load_history()
        self.history_list.delete(0, "end")
        for item in self.history_items:
            label = f"{item['timestamp']} | {item['verdict']} | {item['ai_probability']:.1f}% AI"
            self.history_list.insert("end", label)

    def load_selected_history(self, _event=None) -> None:
        if not self.history_list.curselection():
            return
        index = self.history_list.curselection()[0]
        self.current_result = self.history_items[index]
        self.current_source_label = self.current_result.get("source_label", "History Entry")
        self.ai_prob_var.set(f"{self.current_result['ai_probability']:.2f}%")
        self.human_prob_var.set(f"{self.current_result['human_probability']:.2f}%")
        self.verdict_var.set(self.current_result["verdict"])
        self.summary_var.set(self.current_result["summary"])
        self._refresh_charts()


if __name__ == "__main__":
    app = AIContentDetectorApp()
    app.mainloop()

