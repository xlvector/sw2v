import sys

fname = sys.argv[1]

vocab = {}
sw = open(fname + ".ints", "w")
with open(fname) as f:
    buf = f.read()
    tks = buf.split(' ')
    for tk in tks:
        tk = tk.strip()
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = [len(vocab), 1]
        else:
            vocab[tk][1] += 1
        sw.write(str(vocab[tk][0]) + " ")
sw.close()

sw = open(fname + ".vocab", "w")
sw.write(str(len(vocab)) + "\n")
for k, v in vocab.items():
    sw.write(k + "\t" + str(v[0]) + "\t" + str(v[1]) + "\n")
sw.close()
