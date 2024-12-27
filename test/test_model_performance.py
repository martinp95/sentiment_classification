import unittest
import sys
import os
import time

# Add the path to the source code directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sentiment_classification.sentiment_model import SentimentModel

class TestSentimentModelPerformance(unittest.TestCase):
    """
    Compare DistilBERT vs. BERT performance on a small batch.
    """
    def test_compare_distilbert_vs_bert(self):
        texts = [
            "I love this!",
            "This is terrible.",
            "It could be better, but it's not bad.",
            "Absolutely wonderful experience.",
            "I will never use this again."
        ]

        # Number of times we'll run each model to average performance
        runs = 3

        # DistilBERT timing
        distil_times = []
        for _ in range(runs):
            start = time.perf_counter()
            distil_model = SentimentModel(
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device="cpu"
            )
            _ = distil_model.classify_batch(texts)
            distil_times.append(time.perf_counter() - start)
        distil_avg_time = sum(distil_times) / len(distil_times)

        # BERT timing
        bert_times = []
        for _ in range(runs):
            start = time.perf_counter()
            bert_model = SentimentModel(
                model="bert-base-uncased",
                device="cpu"
            )
            _ = bert_model.classify_batch(texts)
            bert_times.append(time.perf_counter() - start)
        bert_avg_time = sum(bert_times) / len(bert_times)

        print(
            f"\nDistilBERT average time ({runs} runs): {distil_avg_time:.4f} seconds "
            f"for batch of {len(texts)}"
        )
        print(
            f"BERT average time   ({runs} runs): {bert_avg_time:.4f} seconds "
            f"for batch of {len(texts)}\n"
        )

if __name__ == '__main__':
    unittest.main()