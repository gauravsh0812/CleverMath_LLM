import torch

ls = ('10\n', '3\n', '2\n', '5\n')
lbls = []
for _l in ls:
    _l = int(_l.replace("\n",""))
    z = torch.zeros(10)
    z[:_l] = 1
    lbls.append(z)
print(lbls)