import urllib.request
import bs4 as bs

data = urllib.request.urlopen('https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial')

data = data.read()
html_data = bs.BeautifulSoup(data, features='html')

paragraphs = html_data.find_all('p')

content = ''
for paragraph in paragraphs:
    content += paragraph.text
    
content = content.lower()

print(content)

# Search in texts with spaCy
import spacy

nlp = spacy.load('pt_core_news_sm')

string = 'turing'
search_token = nlp(string)

from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)
matcher.add('SEARCH', None, search_token)

document = nlp(content)
matches = matcher(document)

html_code = f"""<h1>{string.upper()}</h1><p>Found Results: {len(matches)}</p><br>"""

number_of_words = 50

for i in matches:
    begin = i[1] - number_of_words
    if begin < 0:
        begin = 0
    html_code += f'<p>{document[begin : i[2] + number_of_words]}</p>'.replace(string, f'<mark>{string}</mark>')
    html_code += f'<br><br>'

with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html_code)
    
import webbrowser
webbrowser.open('output.html')
