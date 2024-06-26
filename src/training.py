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
    clip_enc=False,
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
        # label: (B,11) --  since total number of classes are 0-10 (one-hot encoded)

        ids = ids.to(device)
        attns = attns.to(device)
        labels = labels.to(device, dtype=torch.long)

        if not clip_enc:
            _imgs = list()
            for im in imgs:
                tnsr = torch.load(f"{data_path}/image_tensors/{int(im.item())}.pt")
                _imgs.append(tnsr)
            
            imgs = torch.stack(_imgs).to(device)
        
        else:
            _imgs = list()
            for im in imgs:
                _i = f"{data_path}/images/{int(im.item())}.png"
                _imgs.append(_i)
            
            imgs = torch.stack(_imgs).to(device)
        
        # setting gradients to zero
        optimizer.zero_grad()

        output = model(
            imgs,
            ids,
            attns,
        )

        print("output shape: ", output.shape)

        # output: (B, 11, 11)
        # labels: (B, 11)
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), 
                         labels.contiguous().view(-1))
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not ddp) or (ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss