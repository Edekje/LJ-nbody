import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

data=pickle.load(open(sys.argv[1],'rb'))

maxmsds=[np.max(graph) for graph in data['MSD']]
TVs = data['TV']
PT = data['PostT']

print(TVs)
print(maxmsds)

low = []
intermediate = []
high = []

for i in range(len(maxmsds)):
    if maxmsds[i] > 500:
        high.append(TVs[i])
    elif maxmsds[i] > 50:
        intermediate.append(TVs[i])
    else:
        low.append(TVs[i])

high = np.array([(b,a) for a,b in high])
intermediate = np.array([(b,a) for a,b in intermediate])
low = np.array([(b,a) for a,b in low])

plt.plot(high[:,0],high[:,1], 'r+')
plt.plot(intermediate[:,0], intermediate[:,1], 'y+')
plt.plot(low[:,0], low[:,1], 'g+')

plt.xlabel("V")
plt.ylabel("T")
plt.show()
