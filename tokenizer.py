from collections import Counter

class SimpleTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.idx2word = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, max_len=20):
        tokens = [self.word2idx.get(w, 3) for w in text.lower().split()]
        tokens = [1] + tokens + [2]
        tokens = tokens[:max_len]
        tokens += [0]*(max_len - len(tokens))
        return tokens

    def decode(self, tokens):
        words = []
        for t in tokens:
            if t == 2: break
            words.append(self.idx2word.get(t, "<UNK>"))
        return " ".join(words)