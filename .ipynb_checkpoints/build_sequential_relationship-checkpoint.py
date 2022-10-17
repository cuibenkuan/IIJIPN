# 顺序关系的建立

from math import log


def build_sequential_relationship(doc_content, window_size):

    # print("The doc_content is:",doc_content)
    words = doc_content.split()
    length = len(words)

    doc_word_id_map = {}  # 单词到序号之间的映射
    for j in range(length):
        doc_word_id_map[words[j]] = j

    windows = []  # 首先提取共有多少个窗口
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

    word_window_freq = {}  # 记录某一个单词在所有窗口中出现的频率，相当于相当于单词a或单词b在全局中同时出现的窗口就数目
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    # print("The word_window_freq is:")
    # print(word_window_freq)

    word_pair_count = {}  # 记录全局出现的单词对的频率,相当于单词a和单词b在全局中同时出现的窗口就数目
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                if window[i] == window[j]:
                    continue

                word_pair_str = str(window[i]) + ';' + str(window[j])
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(window[j]) + ';' + str(window[i])
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    # print("The word_pair_count is:")
    # print(word_pair_count)


    # 为每句话构建图
    row = []  # 行
    col = []  # 列
    weight = []  # 顺序权重

    for key in word_pair_count:
        temp = key.split(';')
        i = str(temp[0])
        j = str(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]  # i出现的滑动窗口个数
        word_freq_j = word_window_freq[j]  # j出现的滑动窗口个数
        pmi = log((1.0 * count / len(windows)) /
                  (1.0 * word_freq_i * word_freq_j / (len(windows) * len(windows))))
        if pmi <= 0:
            continue
        # pmi：使用滑动窗口来描述顺序信息
        row.append(doc_word_id_map[i])  # 基于局部的数字表示
        col.append(doc_word_id_map[j])
        weight.append(pmi)

    return length, row, col, weight
    #return length, row, col, weight, word_pair_count
