from spacy.lang.en import English

nlp = English()

doc = nlp("Hello World! 4")

for token in doc:
    print(token.text)

print("Index: ", [ token.i for token in doc] )
print("Text: ", [ token.text for token in doc] )
print("is_alpha", [ token.is_alpha for token in doc] )
print("like_num: ", [ token.like_num for token in doc] )

import spacy

nlp = spacy.load("en_core_web_sm")