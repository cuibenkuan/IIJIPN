from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm
import nltk
nltk.download('wordnet')
import sys
import re
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2016-10-31', lang='en')



def clean_sen(string):
    string = re.sub(r"[^A-Za-z0-9\'.,:\-]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def is_need_lemmas(tag):
    if tag.startswith('J'):
        return 1
    elif tag.startswith('V'):
        return 1
    elif tag.startswith('N'):
        return 1
    elif tag.startswith('R'):
        return 1
    else:
        return 0



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV



if len(sys.argv) != 2:
	sys.exit("Use: python remove_words.py <dataset>")

datasets = ['constraint']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

word_freq_min=2

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))




doc_content_list = []
f = open('data/corpus/' + dataset + '.txt', 'rb')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))


doc_content_clean = []
for doc_content in tqdm(doc_content_list):
    temp = clean_sen(doc_content)
    doc_content_clean.append(temp.strip())

clean_docss = []
for doc_content in tqdm(doc_content_clean):
    words = nlp.word_tokenize(doc_content)
    doc_words = []
    for word in words:
        if word != "," and word != ":" and word != "."and word != "'"and word != "--"and word != ". . ."and word != "-":
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docss.append(doc_str)

clean_docs = []
len_list = []
for doc_content in tqdm(clean_docss):
    doc_content = doc_content.strip()
    tagged_sent = nlp.pos_tag(doc_content)
    wnl = WordNetLemmatizer()
    doc_words = []
    for tag in tagged_sent:
        islemmas_sent = is_need_lemmas(tag[1])
        if islemmas_sent:
            wordnet_pos = get_wordnet_pos(tag[1])
            word = wnl.lemmatize(tag[0], pos=wordnet_pos)
        else:
            word = tag[0]
        doc_words.append(word)
    doc_str = ' '.join(doc_words).strip().lower()
    clean_docs.append(doc_str)

    words = doc_str.split()
    len_list.append(len(words))


print('sum_doc:', len(len_list), '\n',
      'max_doc_length:', max(len_list), '\n',
      'min_doc_length:', min(len_list), '\n',
      'average_doc_length: {:.2f}'.format(np.mean(len_list)), '\n',
      )

clean_corpus_str = '\n'.join(clean_docs)
f = open('data/corpus/' + dataset + '.clean.txt', 'w')
f.write(clean_corpus_str)
f.close()
print("Remove_words have done")

