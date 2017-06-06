# NLP Classifier

A small test to check how difficult it is to write an intent classifier with an entity extractor.

## Setup

All of the code was only tested using Python 3.6 so I'd recommend using it as well.
Before starting, setup a virtual environment and within that environment, install the dependencies:

```
pip install requirements.txt
```

Next, open python up in a shell and download the necessary [nltk](http://www.nltk.org/) data as follows:
```
python
import nltk
nltk.download()
```

Then you need to download a pretrained GloVe model. I got mine from [here](https://nlp.stanford.edu/projects/glove/).
It seems to work really well.

Finally, to run everything just run the following within the virtual environent:
```
python -m question_answering.main
```
