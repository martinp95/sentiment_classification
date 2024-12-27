import unittest
import sys
import os

# Add the path to the source code directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sentiment_classification.sentiment_model import SentimentModel

class TestSentimentModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize model once for all tests (CPU for simplicity)
        cls.model = SentimentModel(device="cpu")

    def test_classify_positive_text(self):
        text = "I absolutely love this product, it's amazing!"
        label, confidence = self.model.classify_text(text)
        self.assertEqual(label, "Positive", "Expected 'Positive' sentiment.")
        self.assertGreater(confidence, 0.5, "Confidence should be > 0.5.")

    def test_classify_negative_text(self):
        text = "I hate how unreliable this service is."
        label, confidence = self.model.classify_text(text)
        self.assertEqual(label, "Negative", "Expected 'Negative' sentiment.")
        self.assertGreater(confidence, 0.5, "Confidence should be > 0.5.")

    def test_classify_neutral_text(self):
        # The model does not have a "Neutral" label, but let's see how it behaves.
        text = "The product is okay, not the best, not the worst."
        label, confidence = self.model.classify_text(text)
        self.assertEqual(label, "Positive", "Expected 'Positive' sentiment.")
        # We might expect lower confidence here
        print(f"Neutral text classification -> Label: {label}, Confidence: {confidence:.4f}")

    def test_classify_ambiguous_text(self):
        # This sentence has both positive and negative phrasing
        text = "It is absolutely fascinating and horrible at the same time."
        label, confidence = self.model.classify_text(text)
        self.assertEqual(label, "Positive", "Expected 'Positive' sentiment.")
        print(f"Ambiguous text classification -> Label: {label}, Confidence: {confidence:.4f}")

if __name__ == '__main__':
    unittest.main()