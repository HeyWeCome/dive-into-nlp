import numpy as np

text = 'You say goodbye and I say hello.'


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split()

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)  # 单词ID长度
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


corpus, word_to_id, id_to_word = preprocess(text)
print('corpus,单词ID列表为:', corpus)
print('word_to_id,单词到单词ID的字典:', word_to_id)
print('id_to_word,单词ID到单词的字典:', id_to_word)
co_matrix = create_co_matrix(corpus, len(corpus))
print("共现矩阵为:\n", co_matrix)
