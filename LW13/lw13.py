from pathlib import Path
import spacy
from spacy import displacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# # Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. ")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
print("Stop:", [token for token in doc if token.is_stop ])

# Find named entities, phrases and conceptsФ
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)


# cake_mask = np.array(Image.open('plane.jpg'))

# генерируем облако слов
cloud = WordCloud(  contour_width=10, contour_color='#2e3043').generate(text)
# увеличим размер выводимой фигуры
plt.figure(figsize=(45, 25))
plt.imshow(cloud)
plt.axis('off')
plt.show()


svg = displacy.render(doc, style="dep")
output_path = Path("1.svg")
output_path.open("w", encoding="utf-8").write(svg)

for ent in doc.ents:
    print(ent.text, ent.label_)

nlp = spacy.load("ru_core_news_sm")

text = ("Американская (США) транснациональная корпорация в составе холдинга Alphabet, инвестирующая в интернет-поиск, облачные вычисления и рекламные технологии. Google поддерживает и разрабатывает ряд интернет-сервисов и продуктов и получает прибыль в первую очередь от рекламы через свою программу Ads.")
doc = nlp(text)

# Analyze syntax

print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
print("Stop:", [token for token in doc if token.is_stop ])

# Find named entities, phrases and concepts
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)




text = str([token for token in doc if not token.is_stop])
# генерируем облако слов
cloud = WordCloud(  contour_width=10, contour_color='#2e3043').generate(text)
# увеличим размер выводимой фигуры
plt.figure(figsize=(45, 25))
plt.imshow(cloud)
plt.axis('off')
plt.show()


svg = displacy.render(doc, style="dep")
output_path = Path("2.svg")
output_path.open("w", encoding="utf-8").write(svg)

for ent in doc.ents:
    print(ent.text, ent.label_)
