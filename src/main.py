from typing import List
from sentiment_classification.sentiment_model import SentimentModel
from sentiment_classification.sentiment_analyzer import SentimentAnalyzer

def get_input_phrases() -> List[str]:
    """
    Prompts the user to enter phrases for classification.

    Returns:
    --------
    list
        List of phrases entered by the user.
    """
    print("Enter the phrases you want to classify (one per line).")
    print("Enter 'exit' to finish.")
    phrases = []
    while True:
        phrase = input("Phrase: ")
        if phrase.lower() == "exit":
            return None
        if phrase == "":
            break
        phrases.append(phrase)
    return phrases

def main():
    try:
        # Instantiate the model
        model_instance = SentimentModel()

        # Create the analyzer
        analyzer = SentimentAnalyzer(model_instance)

        while True:
            # Get phrases from the user
            phrases_to_classify = get_input_phrases()
            if phrases_to_classify is None:
                print("Exiting...")
                break
            if not phrases_to_classify:
                print("No phrases to classify. Please enter some phrases.")
                continue

            # Classify the phrases
            results = analyzer.analyze_batch(phrases_to_classify)

            # Print the results
            print("Classification results:")
            for phrase, result in zip(phrases_to_classify, results):
                print(f"Phrase: {phrase}")
                print(f"Result: {result}")
                print()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()