from nltk.corpus import stopwords
import nltk

import sys
import re
from stanfordcorenlp import StanfordCoreNLP
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
#nlp = StanfordCoreNLP(r'.\stanford-corenlp-full-2016-10-31', lang='en')
nlp = StanfordCoreNLP(r'/home/featurize/data/stanford-corenlp-full-2016-10-31', lang='en')
# nlp = StanfordCoreNLP(r'/home/featurize/data/stanford-corenlp-full-2016-10-31', lang='en')


def clean_sen(string):#1.清理句子，删除无用的字符，保留暂时有用的字符
    string = re.sub(r"[^A-Za-z0-9\'.,:\-]", " ", string)  # 余留小数点.的意思是为了保存出现的小数；保留逗号，的意思是保存300，000这样的大树；保留引号‘的意思是保留缩写
    string = re.sub(r"\s{2,}", " ", string)  # 将多余的空格合并为一个
    return string.strip()# return string.strip().lower()#全部转为小写

# 判断该单词是否需要还原
def is_need_lemmas(tag):#都是需要还原的
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


# 获取需要还原单词的词性
def get_wordnet_pos(tag):#都是需要还原的
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

datasets = ['R8', 'R52', 'ohsumed', 'mr', 'test','liar']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

word_freq_min=2

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# print("The len is:", len(stop_words))
# print("The english stop_words is:")
# print(stop_words)


#1.获取原始内容
doc_content_list = []#存放内容
f = open('data/corpus/' + dataset + '.txt', 'rb')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))  # 获取到原始的内容

#2.初步清理,去除非法字符
doc_content_clean = []#存放内容
for doc_content in doc_content_list:
    # print("The content1 is:", doc_content)
    temp = clean_sen(doc_content)
    # print("The content2 is:", temp)
    # print()
    doc_content_clean.append(temp.strip())

#3.分词之后再进一步进行清理，并去掉保留的一些字符，
clean_docs = []
for doc_content in doc_content_clean:
    words = nlp.word_tokenize(doc_content)
    # print("The tokens is:", words)
    # print()
    doc_words = []
    for word in words:
        if word != "," and word != ":" and word != "."and word != "'"and word != "--"and word != ". . ."and word != "-":#针对上面第一次clean留下的一些字符进行进一步的处理
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)
clean_corpus_str = '\n'.join(clean_docs)
f = open('data/corpus/' + dataset + '.middle.clean.txt', 'w')
f.write(clean_corpus_str)
f.close()

# truncate = False # whether to truncate long document，是否截断长文档
# MAX_TRUNC_LEN = 350 #文档的最大长度为350


#4.获取处理第一次过后每条新闻的长度
doc_content_list = []#存放内容
min_len = 10000
aver_len = 0
max_len = 0
sum = 0
f = open('data/corpus/' + dataset + '.middle.clean.txt', 'rb')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))#获取到原始的内容
for doc_content in doc_content_list:
    doc_content = doc_content.strip()
    temp = nlp.word_tokenize(doc_content)
    # print("The tokens is:", temp)
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)

    sum = sum + 1

aver_len = 1.0 * aver_len / sum
print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(aver_len))
f.close()



print("5.记录词频")
#5.记录词频
word_freq = {}
for doc_content in doc_content_list:

    doc_content = doc_content.strip()
    tagged_sent = nlp.pos_tag(doc_content)
    wnl = WordNetLemmatizer()

    for tag in tagged_sent:
        islemmas_sent = is_need_lemmas(tag[1])
        if islemmas_sent:
            #print("1")需要还原
            wordnet_pos = get_wordnet_pos(tag[1])
            word_hy=wnl.lemmatize(tag[0], pos=wordnet_pos)
        else:
            #print("0")不需要还原
            word_hy=tag[0]
        if word_hy in word_freq:
            word_freq[word_hy] += 1
        else:
            word_freq[word_hy] = 1

#6.最后预处理一遍句子
clean_docs = []
for doc_content in doc_content_list:
    doc_content = doc_content.strip()
    tagged_sent = nlp.pos_tag(doc_content)#获得了单词的词性
    wnl = WordNetLemmatizer()
    doc_words = []
    for tag in tagged_sent:
        islemmas_sent = is_need_lemmas(tag[1])
        if islemmas_sent:# 需要还原
            wordnet_pos = get_wordnet_pos(tag[1])
            word = wnl.lemmatize(tag[0], pos=wordnet_pos)
        else:# 不需要还原
            word = tag[0]
        #下面这个根据数据集自己调整
        if dataset == 'mr' or dataset =='test' or dataset =='liar':
            doc_words.append(word)
        else:
            if word not in stop_words and word_freq[word] >= word_freq_min:
               doc_words.append(word)

    doc_str = ' '.join(doc_words).strip().lower()
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)
f = open('data/corpus/' + dataset + '.clean.txt', 'w')
f.write(clean_corpus_str)
f.close()
print("Remove_words have done")

