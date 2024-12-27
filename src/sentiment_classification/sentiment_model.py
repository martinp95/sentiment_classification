from typing import List, Tuple
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class SentimentModel:
    """
    Encapsulates loading a DistilBert model and the inference logic
    to classify texts as 'Positive' or 'Negative'.
    """

    def __init__(self,
                 model: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 device: str = None):
        """
        Initializes the DistilBERT tokenizer and model for sequence classification.

        Parameters:
        -----------
        model_name : str
            Hugging Face model name or path.
        device : str
            Device to use ("cpu" or "cuda"). If None, it will be auto-detected.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model)
        self.model = DistilBertForSequenceClassification.from_pretrained(model)
        self.model.to(self.device)

    def classify_text(self, text: str) -> Tuple[str, float]:
        """
        Classifies a text as 'Positive' or 'Negative'.

        Parameters:
        -----------
        text : str
            Text to classify.

        Returns:
        --------
        str
            'Positive' or 'Negative'.
        """
        # Tokenize and prepare tensor inputs
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, padding=True)
        inputs.to(self.device)

        # Inference with no gradient calculation (faster, less memory)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction_idx].item()

        # According to the model, [0] => Negative, [1] => Positive
        labels = ["Negative", "Positive"]
        predicted_label = labels[prediction_idx]

        return predicted_label, confidence

    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Classify a batch (list) of texts in one inference pass.
        
        Parameters:
        -----------
        texts : List[str]
            List of texts to classify.

        Returns:
        --------
        List[Tuple[str, float]]
            List of tuples with the predicted label and confidence for each text.
        """
        # Tokenize all texts together
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                padding=True).to(self.device)

        # Inference with no gradient calculation (faster, less memory)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1)
        # Argmax across each row
        predicted_indices = torch.argmax(probabilities, dim=1).tolist()

        # Convert each prediction index to label and confidence
        results = []
        # According to the model, [0] => Negative, [1] => Positive
        labels = ["Negative", "Positive"]

        for i, idx in enumerate(predicted_indices):
            label = labels[idx]
            confidence = probabilities[i][idx].item()
            results.append((label, confidence))

        return results
