import scipy.sparse as sp


def build_syntactic_relationship(nlp,shuffle_doc_words_list, start, end):
    x_adj1 = []
    for i in range(start, end):
        doc_content = shuffle_doc_words_list[i]
        vocab = nlp.word_tokenize(doc_content)
        length = len(vocab)

        row = []
        col = []
        weight = []

        if length == 1:
            adj1 = sp.csr_matrix((weight, (row, col)), shape=(length, length))
        else:
            doc_word_id_map = {}
            for j in range(length):
                doc_word_id_map[vocab[j]] = j

            res = nlp.dependency_parse(doc_content)

            rela_pair_count_str = {}
            for tuple in res:

                if tuple[0] == 'ROOT':
                    continue
                if tuple[1] == tuple[2]:
                    continue

                left = vocab[int(tuple[1]) - 1]
                right = vocab[int(tuple[2]) - 1]

                word_pair_str = left + ';' + right
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1

            max_count1 = 0.0
            min_count1 = 0.0
            for key in rela_pair_count_str:
                if rela_pair_count_str[key] > max_count1:
                    max_count1 = rela_pair_count_str[key]
                if rela_pair_count_str[key] < min_count1:
                    min_count1 = rela_pair_count_str[key]

            row = []
            col = []
            weight = []
            for key in rela_pair_count_str:
                temp = key.split(';')
                i = str(temp[0])
                j = str(temp[1])
                row.append(doc_word_id_map[i])
                col.append(doc_word_id_map[j])
                temp = (rela_pair_count_str[key] - min_count1) / (max_count1 - min_count1)
                weight.append(temp)
            adj1 = sp.csr_matrix((weight, (row, col)), shape=(length, length))

        x_adj1.append(adj1)

    return x_adj1


