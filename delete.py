import torchtext.vocab
import pickle
with open('./rand.pickle', 'rb') as f:
    tmp = pickle.load(f)

with open('./utilities/music_vocab.pkl', 'rb') as f:
    vcb = pickle.load(f)
itos = vcb.get_itos()
x = [itos[tmp[i]] for i in range(len(tmp))]
for i, idx in enumerate(tmp):
    if i == 25:
        print("=" * 25)
    print(itos[idx])
# print(x)
