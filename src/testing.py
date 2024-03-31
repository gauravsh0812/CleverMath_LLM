# -*- coding: utf-8 -*-

import torch

def evaluate(
    model,
    data_path,
    test_dataloader,
    criterion,
    device,
    lmm_enc=False,
    is_test=False,
):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (imgs, ids, attns, labels) in enumerate(test_dataloader):
            ids = ids.to(device)
            attns = attns.to(device)
            labels = labels.to(device, dtype=torch.long)
        
            _imgs = list()
            for im in imgs:
                tnsr = torch.load(f"{data_path}/image_tensors/{int(im.item())}.pt")
                _imgs.append(tnsr)
            imgs = torch.stack(_imgs).to(device)

            output = model(imgs,ids,attns)

            loss = criterion(
                            output.contiguous().view(-1,output.shape[-1]), 
                            labels.contiguous().view(-1)
                            )

            epoch_loss += loss.item()
            
            accuracy = None
            if is_test:
                # output: (B, 11, 11)
                # labels: (B, 11)
                test_labels = open("logs/test_labels.txt", "w")
                test_preds = open("logs/test_preds.txt", "w")
                N = output.shape[0]
                correct = 0
                for b in range(N):
                    zl = labels[b,:]
                    lbl = [i for i in range(len(zl)) if zl[i]==1.0][0]

                    zo = output[b,-1,:] # last time step (11)
                    pred = torch.argmax(zo,dim=0).item()

                    test_labels.write(str(lbl) + "\n")
                    test_preds.write(str(pred) + "\n")

                    if int(pred) == int(lbl): correct+=1

                accuracy = correct / N

    net_loss = epoch_loss / len(test_dataloader)
    return net_loss, accuracy