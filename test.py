import sys, math
from operator import itemgetter

def sim(v1, v2):
    ret = 0.0
    for i in range(len(v1)):
        ret += v1[i] * v2[i]
    return ret

w1 = sys.argv[1]

model = {}
for line in file("./data/text8_sw2v.model"):
    tks = line.strip().split(' ')
    model[tks[0]] = [float(x) for x in tks[1:]]

f1 = model[w1]
w = {}
for k, f2 in model.items():
    if len(f1) != len(f2):
        print k
        continue
    w[k] = sim(f1, f2)

for k, v in sorted(w.items(), key = itemgetter(1), reverse = True)[:20]:
    print k, v
