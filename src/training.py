# -*- coding: utf-8 -*-
import torch
from tqdm.auto import tqdm

def train(
    model,
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

    for i, (img, qtn, label) in enumerate(tset):
        # qtn: (B, len)  --> the padding will be done in the RobertaEncoder
        # img: (B, in_channel, H, W)
        # label: (B,)

        qtn = qtn.to(device, dtype=torch.long)
        
        _imgs = list()
        for im in img:
            # for vit patch encoder 
            _imgs.append(torch.load(f"data/image_tensors/{int(im.item())}.pt"))

            # for pixel encoders
            # for vit pixel encoder, _imgs will be same
            if isGraphPixel:
                G = torch.load(f"{img_graph_path}/{int(im.item())}.pt")
                _data_list.append(G)
        
        if isGraphPixel:
            batch = Batch.from_data_list(_data_list).to(device)
        else:
            batch=None

        imgs = torch.stack(_imgs).to(device)
        
        # setting gradients to zero
        optimizer.zero_grad()

        outputs, _ = model(
            imgs,
            batch,
            mml,
        )
        output_dim = outputs.shape[-1]

        # avoiding <sos> token while Calculating loss
        mml = mml[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, output_dim)

        loss = criterion(outputs, mml)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not ddp) or (ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss