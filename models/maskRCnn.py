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

class MaskRCNN(nn.Module):
    def __init__(self,):
        super(MaskRCNN, self).__init__()
        self.cnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.cnn.eval()
    
    def forward(self,_x, im):
        _x = self.cnn([_x])    
        scores = _x[0]['scores']
        masks = _x[0]['masks']      # (n_masks, 1, w,h)
        print(masks.shape)
        torch.save(scores, f"{cfg.dataset.path_to_data}/maskrcnn/scores/{im}")
        torch.save(masks, f"{cfg.dataset.path_to_data}/maskrcnn/masks/{im}")

        # print("masks shape: ", masks.shape)
            
        # top_masks = []

        # if masks.shape[0] >= self.top_n:
        #     top_n_scores = sorted(scores, reverse=True)[:self.top_n]
        #     for _s in top_n_scores:
        #         top_n_index = scores.index(_s)
        #         top_n_mask = masks[top_n_index,:,:,:] #(1,1,w,h)
        #         top_masks.append(top_n_mask)        
        #     masks = torch.stack(top_masks).squeeze(1)  # (top_n,w,h)
        #     final_masks.append(masks)
        #     # print("stacked masks shape: ", masks.shape)

        # else:
        #     delta = self.top_n - masks.shape[0] 
        #     top_n_scores = sorted(scores, reverse=True)[:self.top_n]
        #     for _s in top_n_scores:
        #         top_n_index = scores.index(_s)
        #         top_n_mask = masks[top_n_index,:,:,:] #(1,1,w,h)
        #         top_masks.append(top_n_mask)      

        #     masks = torch.stack(top_masks).squeeze(1)  # (mask_shape[0], 1, w,h)          
        #     zeros = torch.zeros(delta,masks.shape[-2],masks.shape[-1])  
        #     masks = torch.cat((masks, zeros), dim=0)  #(top_n,1,w,h)
        #     final_masks.append(masks)
        #     # print("stacked 2 masks shape: ", masks.shape)
        
        # final_masks = torch.stack(final_masks)
        # return masks # (B, n_masks, w,h)


def main():
    m = MaskRCNN()
    tnsrs = os.listdir(f"{cfg.dataset.path_to_data}/image_tensors")
    
    os.mkdir(f"{cfg.dataset.path_to_data}/maskrcnn/")
    os.mkdir(f"{cfg.dataset.path_to_data}/maskrcnn/masks")
    os.mkdir(f"{cfg.dataset.path_to_data}/maskrcnn/scores")

    for im in tnsrs:
        _i = torch.load(f"{cfg.dataset.path_to_data}/image_tensors/{im}")[1:]
        y = m([_i], im)
        # plot(y)

# def plot(y):
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

if "__init__" == "__main__":
    main()
