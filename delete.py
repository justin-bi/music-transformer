import pickle
from torchtext.vocab import build_vocab_from_iterator


with open('./music_vocab.pkl', 'rb') as f:
    vcb = pickle.load(f)
    print(vcb.get_itos())
    print(len(vcb))
