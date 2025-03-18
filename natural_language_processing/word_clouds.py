from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import urllib.request
import bs4 as bs
import spacy

nlp = spacy.load('pt_core_news_sm')

data = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')

data = data.read()
html_data = bs.BeautifulSoup(data, features='html.parser')

paragraphs = html_data.find_all('p')

content = ''
for paragraph in paragraphs:
    content += paragraph.text
    
content = content.lower()

document = nlp(content)
token_list = []

for token in document:
    if nlp.vocab[token.text].is_stop == False:
        token_list.append(token.text)

color_map = ListedColormap(['orange', 'green', 'red', 'magenta'])

cloud = WordCloud(background_color='white', max_words=100, colormap=color_map)

cloud = cloud.generate(''.join(token_list))
plt.figure(figsize=(15, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()

