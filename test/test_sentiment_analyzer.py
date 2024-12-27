import unittest
import sys
import os

# Add the path to the source code directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sentiment_classification.sentiment_model import SentimentModel
from sentiment_classification.sentiment_analyzer import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load model and create analyzer
        model = SentimentModel(device="cpu")
        cls.analyzer = SentimentAnalyzer(model)

    def test_analyze_positive_text(self):
        text = "This is the best thing that has ever happened to me!"
        result_str = self.analyzer.analyze_text(text)
        self.assertIn("Positive", result_str, "Expected 'Positive' in the output string.")
        self.assertIn("confidence", result_str, "Expected 'confidence' in the output string.")
        self.assertGreater(float(result_str.split()[-1]), 0.5, "Confidence should be > 0.5.")

    def test_analyze_negative_text(self):
        text = "I am extremely disappointed and unsatisfied."
        result_str = self.analyzer.analyze_text(text)
        self.assertIn("Negative", result_str, "Expected 'Negative' in the output string.")
        self.assertIn("confidence", result_str, "Expected 'confidence' in the output string.")
        self.assertGreater(float(result_str.split()[-1]), 0.5, "Confidence should be > 0.5.")

    def test_analyze_neutral_text(self):
        text = "The product is okay, not the best, not the worst."
        result_str = self.analyzer.analyze_text(text)
        self.assertIn("Positive", result_str, "Expected 'Positive' in the output string.")
        self.assertIn("confidence", result_str, "Expected 'confidence' in the output string.")
        print(f"Neutral text classification -> {result_str}")

    def test_analyze_ambiguous_text(self):
        text = "It is absolutely fascinating and horrible at the same time."
        result_str = self.analyzer.analyze_text(text)
        self.assertIn("Positive", result_str, "Expected 'Positive' in the output string.")
        self.assertIn("confidence", result_str, "Expected 'confidence' in the output string.")
        print(f"Ambiguous text classification -> {result_str}")

    def test_analyze_batch(self):
        texts = [
            "I love this!",
            "This is terrible.",
            "It's okay, I guess."
        ]
        results = self.analyzer.analyze_batch(texts)
        self.assertEqual(len(results), 3, "Should return a result string for each input text.")
        for res in results:
            self.assertIn("Sentiment:", res, "Each result should contain 'Sentiment:'.")
            self.assertIn("confidence", res, "Each result should contain 'confidence'.")
            self.assertGreater(float(res.split()[-1]), 0.5, "Confidence should be > 0.5.")

if __name__ == '__main__':
    unittest.main()