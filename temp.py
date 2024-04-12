import torch, os

_p = "/groups/claytonm/gauravs_data/clevrmath_data/maskrcnn/masks"

nos = open("logs/no_mask_file.lst","w")

for i in os.listdir(_p):
    _i = os.path.join(_p,i)
    n = torch.load(_i).shape[0]
    if n == 0:
        print(i)
        nos.write(f"{i} \n")