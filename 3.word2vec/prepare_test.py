import numpy as np

from common.util import preprocess, convert_one_hot

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)


def create_contexts_target(corpus, window_size=1):
    # 目标词是除去首位两位的中间句子 [X, target, target, ..., target, X]
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


contexts, target = create_contexts_target(corpus)

# 处理为one-hot编码
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

print('contexts:', contexts.shape)
print(contexts)
print('target:')
print(target)