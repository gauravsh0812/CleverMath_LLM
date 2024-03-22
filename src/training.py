# -*- coding: utf-8 -*-
import torch
from tqdm.auto import tqdm

def train(
    model,
    data_path, 
    train_dataloader,
    optimizer,
    criterion,
    clip,
    device,
    ddp=False,
    rank=None,
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    tset = tqdm(iter(train_dataloader))

    for i, (imgs, ids, attns, labels) in enumerate(tset):
        # ids (qtn input ids from tokenizer): (B, max_len) after padding by tokenizer
        # attn: qtn_attn_mask for padding by tokenizer: (B, max_len)
        # img: (B, in_channel, H, W)
        # label: (B,10) --  since total number of classes are 0-10

        ids = ids.to(device)
        attns = attns.to(device)
        labels = labels.to(device, dtype=torch.long)
        
        _imgs = list()
        for im in imgs:
            tnsr = torch.load(f"{data_path}/image_tensors/{int(im.item())}.pt")
            _imgs.append(tnsr)
            
        
        imgs = torch.stack(_imgs).to(device)
        
        # setting gradients to zero
        optimizer.zero_grad()

        output = model(
            imgs,
            ids,
            attns,
        )
        print("output shape: ", output.shape)
        exit()

        loss = criterion(output, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not ddp) or (ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss