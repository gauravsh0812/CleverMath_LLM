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

    with torch.no_grad():
        for i, (imgs, ids, attns, labels) in enumerate(test_dataloader):
            ids = ids.to(device)
            attns = attns.to(device)
            labels = labels.to(device, dtype=torch.long)
        
            _imgs = list()
            for im in imgs:
                _i = f"{data_path}/images/{int(im.item())}.png"
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
            accuracy += len([i for i in range(len(p)) if p[i] == l[i]])
            
    net_loss = epoch_loss / len(test_dataloader)
    accuracy = accuracy / len(test_dataloader)
    return net_loss, accuracy, pred_labels, labels