
def display_topics(keywords_dic, label, model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):

        klist = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]

        for k in klist:
            if not k in keywords_dic:
                keywords_dic[k] = []
            keywords_dic[k].append(label)
    return keywords_dic


def build_semantic_relationship(content_list, label_list, start, end):
    label = []
    content = []
    label_set = set()
    for i in range(start, end):
        content.append(content_list[i].strip())
        temp = label_list[i].strip()
        temp = temp.split("\t")
        label.append(temp[2])
        label_set.add(temp[2])
    label_length = len(label_set)

    label_set = list(label_set)

    keywords_dic = {}

    for i in range(label_length):
        label_sen = label_set[i]

        doc_content_list = []
        for j in range(len(content)):
            if label[j] == label_sen:
                doc_content_list.append(content[j].strip())

        keywords_dic = display_topics(keywords_dic, label_sen,len(doc_content_list))


    word_pair_count = {}
    for i in range(len(content)):
        doc_words = content[i]
        doc_words = doc_words.strip()
        words = doc_words.split()
        for word in words:
            if word in keywords_dic:
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
