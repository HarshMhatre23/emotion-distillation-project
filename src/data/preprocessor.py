import numpy as np
def tokenize_dataset(dataset, tokenizer, max_length=128):
    def fn(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=max_length)
    return dataset.map(fn, batched=True)

def multi_hot(labels_list, n=27):
    out = np.zeros((len(labels_list), n), dtype=np.float32)
    for i, ls in enumerate(labels_list):
        for l in ls:
            if l < n: out[i, l] = 1.0
    return out
