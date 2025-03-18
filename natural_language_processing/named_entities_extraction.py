import urllib.request
import bs4 as bs
import spacy

data = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')

data = data.read()
html_data = bs.BeautifulSoup(data, features='html.parser')

paragraphs = html_data.find_all('p')

content = ''
for paragraph in paragraphs:
    content += paragraph.text
    
content = content.lower()

nlp = spacy.load('pt_core_news_sm')
document = nlp(content)

for entity in document.ents:
    print(entity.text, entity.label_)
    
from spacy import displacy
    
html_code = displacy.render(document, style='ent')

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_code)
    
import webbrowser
webbrowser.open('output.html')

