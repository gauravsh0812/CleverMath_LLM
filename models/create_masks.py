import torch, torchvision
import torch.nn as nn
import yaml
from box import Box
import os
# import numpy as np 
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

with open("config/config.yaml") as f:
        cfg = Box(yaml.safe_load(f))

no_masks_file = open("logs/no_mask_images.lst","w")

class MaskRCNN(nn.Module):
    def __init__(self,):
        super(MaskRCNN, self).__init__()
        self.cnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.cnn.eval()
    
    def forward(self,_x, im):
        _x = self.cnn(_x)    
        scores = _x[0]['scores']
        masks = _x[0]['masks']      # (n_masks, 1, w,h)
        if masks.shape[0] > 0:
            torch.save(scores, f"{cfg.dataset.path_to_data}/maskrcnn/scores/{im}")
            torch.save(masks, f"{cfg.dataset.path_to_data}/maskrcnn/masks/{im}")
        else:
            no_masks_file.write(f"{im} \n")    
            
m = MaskRCNN()
tnsrs = os.listdir(f"{cfg.dataset.path_to_data}/image_tensors")

paths = [f"{cfg.dataset.path_to_data}/maskrcnn/",
         f"{cfg.dataset.path_to_data}/maskrcnn/masks",
         f"{cfg.dataset.path_to_data}/maskrcnn/scores"]
for _p in paths:
     if not os.path.exists(_p):
        os.mkdir(_p)

for im in tnsrs:
    _i = torch.load(f"{cfg.dataset.path_to_data}/image_tensors/{im}")[1:]
    m([_i], im)
    

    


#     # Visualize the results
#     fig, ax = plt.subplots(1)
#     ax.imshow(plt.imread("data/images/200.png"))

#     # Draw bounding boxes
#     print("box shape: ", y['boxes'].shape)
#     for box in y['boxes']:
#         box = box.detach().numpy()
#         rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)


#     # Draw masks
#     print("masks shape: ", y['masks'].shape)
#     for mask in y['masks']:
#         mask = mask.detach().numpy()
#         # mask = mask.squeeze().numpy()
#         mask = np.where(mask > 0.5, 1, 0) # Thresholding the mask
#         ax.imshow(mask, cmap='gray', alpha=0.5)

#     plt.show()
