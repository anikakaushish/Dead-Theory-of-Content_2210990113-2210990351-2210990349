# AI Content Detector

AI Content Detector is a desktop application built with Python and Tkinter for estimating whether a piece of writing looks more AI-generated or human-written.

## Features

- Modern Tkinter interface with dark and light themes
- Analyze pasted text or uploaded TXT, DOCX, PDF, and image files
- Hybrid scoring pipeline using:
  - Transformer-based perplexity with `distilgpt2`
  - Burstiness
  - Repetition analysis
  - Sentence length variance
  - Vocabulary richness
- Final confidence score produced by a context-aware heuristic scoring pipeline
- Context-aware confidence damping for short or prompt-like samples
- Pie chart and bar chart visualizations using `matplotlib`
- Export to TXT and PDF
- Copy report to clipboard
- Persistent local scan history in `history.json`

## Project Structure

```text
ai_content_detector/
│── main.py
│── detector.py
│── utils.py
│── requirements.txt
│── README.md
│── assets/
```

## Installation

```bash
pip install -r requirements.txt
python main.py
```

## Notes

- The first transformer-based run may download model weights for `distilgpt2`.
- Image uploads use OCR when Tesseract is installed and available on your system.
- The detector combines heuristic and ML signals, so it should be treated as an assistive estimate rather than a definitive classifier.
- Very short, repetitive question lists may return an inconclusive result because they do not provide enough writing context.
