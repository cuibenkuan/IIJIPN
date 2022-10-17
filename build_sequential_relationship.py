from math import log


def build_sequential_relationship(doc_content, window_size):


    words = doc_content.split()
    length = len(words)

    doc_word_id_map = {}
    for j in range(length):
        doc_word_id_map[words[j]] = j

    windows = []
    if length <= window_size:
        windows.append(words)
    else:

        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

    word_window_freq = {}
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

    word_pair_count = {}
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


    row = []
    col = []
    weight = []

    for key in word_pair_count:
        temp = key.split(';')
        i = str(temp[0])
        j = str(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = log((1.0 * count / len(windows)) /
                  (1.0 * word_freq_i * word_freq_j / (len(windows) * len(windows))))
        if pmi <= 0:
            continue

        row.append(doc_word_id_map[i])
        col.append(doc_word_id_map[j])
        weight.append(pmi)

    return length, row, col, weight

