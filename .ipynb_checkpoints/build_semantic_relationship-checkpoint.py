# 语义关系的建立，利用LDA来找到每一句话的主题分布
# 顺序关系的建立
import sys
import pickle
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def display_topics(keywords_dic, label, model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        # print("Topic %d:" % (topic_idx))
        klist = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        # print(" ".join(klist))
        for k in klist:
            if not k in keywords_dic:
                keywords_dic[k] = []
            keywords_dic[k].append(label)
    return keywords_dic


def build_semantic_relationship(content_list, label_list, start, end):
    label = []  # 获取标签，包含每一条新闻的标签
    content = []  # 内容
    label_set = set()  # 里面仅有标签的种类：0，1
    for i in range(start, end):
        content.append(content_list[i].strip())  # 内容
        temp = label_list[i].strip()
        temp = temp.split("\t")
        label.append(temp[2])
        label_set.add(temp[2])
    label_length = len(label_set)  # 为set类型

    label_set = list(label_set)  # 转list

    keywords_dic = {}  # 关于主题分布的关键词

    for i in range(label_length):
        label_sen = label_set[i]
        doc_content_list = []
        for j in range(len(content)):
            if label[j] == label_sen:
                doc_content_list.append(content[j].strip())

        vectorizer = CountVectorizer(stop_words='english', max_df=0.98)

        # CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
        # CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数。
        # max_df: 可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。如果参数中已经给定了vocabulary，则这个参数无效

        vector = vectorizer.fit_transform(doc_content_list)  # 拟合模型，并返回文本矩阵，fit_transform是fit和transform的组合，既包括了训练又包含了转换
        # print("65 The vector is :", vector)

        feature_names = vectorizer.get_feature_names()
        # print("68 The feature_names is :", feature_names)  # 获取到的关于主题相关的词语

        lda = LatentDirichletAllocation(n_components=1, learning_method='online', learning_offset=50.,
                                        random_state=0).fit(vector)
        # print("71 The lda is", lda)

        keywords_dic = display_topics(keywords_dic, label_sen, lda, feature_names,
                                      len(doc_content_list))  # 调用上面的一个函数，来展示与当前主题最相关的前N个单词,这人的N,我把他设置为数据集或训练集或测试集的数目大小，这儿可以修改，选择统计出来多少单词

    # print(len(keywords_dic))
    # print("keywords_dic is:")
    # print()
    # print(keywords_dic)

    word_pair_count = {}
    for i in range(len(content)):
        # print("the i is:", i)
        doc_words = content[i]
        doc_words = doc_words.strip()
        words = doc_words.split()
        for word in words:
            if word in keywords_dic:
                # print("79", type(label[i]))
                # print("80", type(keywords_dic[word]))
                if label[i] in keywords_dic[word]:
                    for word_n in words:
                        if word != word_n:
                            word_pair_str = str(word) + ';' + str(word_n)
                            if word_pair_str in word_pair_count:
                                word_pair_count[word_pair_str] += 1
                            else:
                                word_pair_count[word_pair_str] = 1

    max_count1 = 0.0
    min_count1 = 0.0
    for key in word_pair_count:
        if word_pair_count[key] > max_count1:
            max_count1 = word_pair_count[key]
        if word_pair_count[key] < min_count1:
            min_count1 = word_pair_count[key]

    return word_pair_count, max_count1, min_count1
