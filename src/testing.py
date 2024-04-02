# -*- coding: utf-8 -*-

import torch

def evaluate(
    model,
    data_path,
    test_dataloader,
    criterion,
    device,
    is_test=False,
):
    model.eval()
    epoch_loss = 0
    accuracy = 0

    if is_test:
        labels_file = open("logs/lables.txt","w")
        labels_file.write("True \t Pred \n")

    with torch.no_grad():
        for i, (imgs, ids, attns, labels) in enumerate(test_dataloader):
            ids = ids.to(device)
            attns = attns.to(device)
            labels = labels.to(device, dtype=torch.long)
        
            _imgs = list()
            for im in imgs:
                _i = f"{data_path}/image_tensors/{int(im.item())}.pt"
                _imgs.append(_i)

            output = model(_imgs,
                           ids,
                           attns,
                           device)
            
            labels = torch.argmax(labels, dim=1)
            loss = criterion(
                            output.contiguous().view(-1,output.shape[-1]), 
                            labels.contiguous().view(-1)
                            )

            epoch_loss += loss.item()
            
            
            pred_labels = torch.argmax(output, dim=1)
            l = labels.cpu().tolist()
            p = pred_labels.cpu().tolist()
            
            count=0
            for i in range(len(p)):
                if p[i] == l[i]:
                    count+=1
                
                if is_test:
                    labels_file.write(f"{l[i]} \t\t {p[i]} \n")

            accuracy+= count / len(p)    

    net_loss = epoch_loss / len(test_dataloader)
    accuracy = accuracy / len(test_dataloader)
    return net_loss, accuracy