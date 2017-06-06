from question_answering.rnn_classifier.classify import Classifier
from question_answering.entity_extraction import extraction

from pprint import PrettyPrinter

def main():
    classifier = Classifier()
    pp = PrettyPrinter(indent=2)

    while True:
        raw_text = input("Ask me a question: ")

        intent = classifier.predict(raw_text)
        print("Intent:")
        pp.pprint(intent)

        entities = extraction.extraction(raw_text)
        print("Entities:")
        pp.pprint(entities)

if __name__ == '__main__':
    main()
