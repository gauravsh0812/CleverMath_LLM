import torch
lbls = [0, 1, 2, 3, 4, 5, 6]
for _l in lbls:
    _l = int(_l)
    z = torch.zeros(11)
    z[_l] = 1.0
    print(z)
    
    print([i for i in range(len(z)) if z[i]==1.0][0])