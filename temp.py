import torch, os

_p = "/groups/claytonm/gauravs_data/clevrmath_data/maskrcnn/masks"

nos = []

for i in os.listdir(_p):
    _i = os.path.join(_p,i)
    n = torch.load(_i).shape[0]
    if n == 0:
        print(i)
        nos.append(i)
print(f" ====================>>> {len(nos)}")