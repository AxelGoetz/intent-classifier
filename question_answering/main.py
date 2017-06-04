from rnn_classifier.classifier import Classifier
from entity_extraction import extraction

def main():
    classifier = Classifier()

    while True:
        raw_text = input("Ask me a question: ")

        intent = classifier.predict(raw_text)
        print("Intent: " + str(intent))

        entities = extraction.extraction(raw_text)
        print("Entities: " + str(entities))
