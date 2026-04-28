import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import nltk
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class DetectionResult:
    ai_probability: float
    human_probability: float
    verdict: str
    metrics: Dict[str, float]
    summary: str


class AIContentDetector:
    def __init__(self) -> None:
        self._ensure_nltk()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "distilgpt2"
        self.tokenizer = None
        self.language_model = None
        self.model_status = "Transformer model unavailable, using statistical fallback."
        self._load_language_model()

    def _ensure_nltk(self) -> None:
        packages = ["punkt", "punkt_tab"]
        for package in packages:
            try:
                if package == "punkt":
                    nltk.data.find("tokenizers/punkt")
                else:
                    nltk.download(package, quiet=True)
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass

    def _load_language_model(self) -> None:
        try:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            self.language_model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
            self.language_model.eval()
            self.model_status = f"Loaded {self.model_name} for perplexity scoring."
        except Exception:
            self.tokenizer = None
            self.language_model = None

    def analyze_text(self, text: str) -> DetectionResult:
        cleaned_text = self._normalize_text(text)
        if len(cleaned_text) < 40:
            raise ValueError("Please provide more text so the detector has enough language context.")

        sentences = sent_tokenize(cleaned_text)
        words = [token.lower() for token in word_tokenize(cleaned_text) if any(ch.isalpha() for ch in token)]
        if len(sentences) < 2 or len(words) < 12:
            raise ValueError("The text is too short for a stable analysis. Add at least a few sentences.")

        perplexity = self._calculate_perplexity(cleaned_text, words)
        burstiness = self._calculate_burstiness(sentences)
        repetition = self._calculate_repetition(words)
        sentence_variance = self._calculate_sentence_variance(sentences)
        vocabulary_richness = self._calculate_vocabulary_richness(words)

        metrics = {
            "Perplexity": perplexity,
            "Burstiness": burstiness,
            "Repetition": repetition,
            "Sentence Variance": sentence_variance,
            "Vocabulary Richness": vocabulary_richness,
        }

        probability = self._score_ai_probability(metrics)
        probability = self._apply_context_adjustment(probability, cleaned_text, words, sentences)

        ai_probability = round(probability * 100, 2)
        human_probability = round((1 - probability) * 100, 2)
        verdict = self._determine_verdict(ai_probability)
        summary = self._build_summary(metrics, verdict, cleaned_text, words, sentences)

        return DetectionResult(
            ai_probability=ai_probability,
            human_probability=human_probability,
            verdict=verdict,
            metrics={key: round(value, 4) for key, value in metrics.items()},
            summary=summary,
        )

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text

    def _calculate_perplexity(self, text: str, words: List[str]) -> float:
        if self.language_model is not None and self.tokenizer is not None:
            try:
                encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                with torch.no_grad():
                    output = self.language_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    loss = float(output.loss.detach().cpu())
                return min(math.exp(loss), 250.0)
            except Exception:
                pass

        frequencies = Counter(words)
        total = len(words)
        entropy = -sum((count / total) * math.log2(count / total) for count in frequencies.values())
        return float(min(250.0, max(5.0, 15 + entropy * 10)))

    def _calculate_burstiness(self, sentences: List[str]) -> float:
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences if sentence.strip()]
        mean_length = statistics.mean(sentence_lengths)
        stdev = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
        return float(max(0.0, min(1.0, stdev / (mean_length + 1))))

    def _calculate_repetition(self, words: List[str]) -> float:
        if len(words) < 4:
            return 0.0
        bigrams = list(zip(words, words[1:]))
        counts = Counter(bigrams)
        repeated = sum(count for count in counts.values() if count > 1)
        return float(max(0.0, min(1.0, repeated / max(len(bigrams), 1))))

    def _calculate_sentence_variance(self, sentences: List[str]) -> float:
        lengths = [len(word_tokenize(sentence)) for sentence in sentences if sentence.strip()]
        if len(lengths) < 2:
            return 0.0
        variance = statistics.variance(lengths)
        return float(max(0.0, min(1.0, variance / 120)))

    def _calculate_vocabulary_richness(self, words: List[str]) -> float:
        unique_ratio = len(set(words)) / max(len(words), 1)
        long_word_bonus = sum(1 for word in words if len(word) > 6) / max(len(words), 1)
        return float(max(0.0, min(1.0, (unique_ratio * 0.75) + (long_word_bonus * 0.25))))

    def _normalize_perplexity(self, perplexity: float) -> float:
        score = 1 - min(max((perplexity - 10) / 90, 0.0), 1.0)
        return float(score)

    def _score_ai_probability(self, metrics: Dict[str, float]) -> float:
        weighted_score = (
            (self._normalize_perplexity(metrics["Perplexity"]) * 0.30)
            + ((1 - metrics["Burstiness"]) * 0.20)
            + (metrics["Repetition"] * 0.20)
            + ((1 - metrics["Sentence Variance"]) * 0.15)
            + ((1 - metrics["Vocabulary Richness"]) * 0.15)
        )
        return float(max(0.0, min(1.0, weighted_score)))

    def _apply_context_adjustment(
        self,
        probability: float,
        text: str,
        words: List[str],
        sentences: List[str],
    ) -> float:
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences if sentence.strip()]
        average_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0.0
        question_ratio = text.count("?") / max(len(sentences), 1)

        word_factor = min(max((len(words) - 20) / 80, 0.0), 1.0)
        sentence_factor = min(max((len(sentences) - 3) / 6, 0.0), 1.0)
        length_factor = min(max((average_sentence_length - 5) / 10, 0.0), 1.0)
        context_factor = (word_factor + sentence_factor + length_factor) / 3

        if question_ratio >= 0.6 and average_sentence_length <= 8:
            context_factor *= 0.35
        elif len(words) < 35:
            context_factor *= 0.6

        context_factor = max(0.0, min(1.0, context_factor))
        return float(0.5 + ((probability - 0.5) * context_factor))

    def _determine_verdict(self, ai_probability: float) -> str:
        if ai_probability >= 65:
            return "AI-generated"
        if ai_probability <= 35:
            return "Human-written"
        return "Inconclusive"

    def _build_summary(
        self,
        metrics: Dict[str, float],
        verdict: str,
        text: str,
        words: List[str],
        sentences: List[str],
    ) -> str:
        reasons = []
        if metrics["Perplexity"] < 35:
            reasons.append("lower perplexity")
        else:
            reasons.append("more varied token probability")
        if metrics["Burstiness"] < 0.25:
            reasons.append("consistent sentence rhythm")
        else:
            reasons.append("more uneven sentence rhythm")
        if metrics["Vocabulary Richness"] > 0.55:
            reasons.append("strong vocabulary variety")
        else:
            reasons.append("limited vocabulary variety")

        question_ratio = text.count("?") / max(len(sentences), 1)
        average_sentence_length = statistics.mean(
            [len(word_tokenize(sentence)) for sentence in sentences if sentence.strip()]
        )

        if question_ratio >= 0.6 and average_sentence_length <= 8:
            reasons.append("prompt-like short questions reduced confidence")
        elif len(words) < 35:
            reasons.append("limited sample size reduced confidence")

        return f"{verdict} based on {', '.join(reasons)}."
