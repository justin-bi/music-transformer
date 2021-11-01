import pickle

with open('./dataset/mel-chords/test/a/a-great-big-world/say-something/chorus_symbol_key.json', 'rb') as f:
    tmp = pickle.load(f)

with open('./music_vocab.pkl', 'rb') as f:
    vcb = pickle.load(f)
    for i in tmp:
        print(vcb.get_itos()[i])
