from typing import List
from sentiment_classification.sentiment_model import SentimentModel

class SentimentAnalyzer:
    """
    Provides a higher-level sentiment analysis service using SentimentModel
    """

    def __init__(self, model: SentimentModel):
        """
        Initializes the SentimentAnalyzer with a SentimentModel.

        Parameters:
        -----------
        model : SentimentModel
            An instance of SentimentModel that handles classification.
        """
        self.model = model

    def analyze_text(self, text: str)->str:
        """
        Returns a string describing the sentiment and confidence.

        Parameters:
        -----------
        text : str
            Text to analyze.
        
        Returns:
        --------
        str
            In the format: "Sentiment: <label>, with confidence <confidence>"
        """
        label, confidence = self.model.classify_text(text)
        return f"Sentiment: {label}, with confidence {confidence:.4f}"
    
    def analyze_batch(self, texts: List[str]) -> List[str]:
        """
        Classify a batch (list) of texts in one pass, returning formatted strings.

        Parameters:
        -----------
        texts : List[str]
            List of texts to analyze.
        
        Returns:
        --------
        List[str]
            List of strings in the format: "Sentiment: <label>, with confidence <confidence>"
        """
        results = self.model.classify_batch(texts)
        output = []
        for (label, conf) in results:
            output.append(f"Sentiment: {label}, with confidence {conf:.4f}")
        return output
