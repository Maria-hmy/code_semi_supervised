#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.metric import dice_coeff
import logging
import os
from torch.autograd import Variable
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
import os
from utils.metrics import dice_loss, dice_coefficient
    
def write_metric_values(net, output, epoch, train_metric, val_metric):
    file = open(output+'epoch-%0*d.txt'%(2,epoch),'w') 
    if net.n_classes > 1:
        logging.info('training cross-entropy: {}'.format(train_metric[-1]))
        logging.info('validation cross-entropy: {}'.format(val_metric[-1]))
        file.write('train cross-entropy = %f\n'%(train_metric[-1]))
        file.write('val cross-entropy = %f'%(val_metric[-1]))
    else:
        logging.info('training dice: {}'.format(train_metric[-1]))
        logging.info('validation dice: {}'.format(val_metric[-1]))
        file.write('train dice = %f\n'%(train_metric[-1]))
        file.write('val dice = %f'%(val_metric[-1]))
    file.close()
    
def display_subloss_values(output, epoch, alpha, subl1e, subl2e):
    file = open(output+'epoch-%0*d-sb.txt'%(2,epoch),'w') 
    logging.info('subl1e: {}'.format(subl1e[-1]))
    logging.info('subl2e: {}'.format(subl2e[-1]))
    logging.info('alpha*subl2e: {}'.format(alpha*subl2e[-1]))
    logging.info('loss: {}'.format(subl1e[-1]+alpha*subl2e[-1]))
    file.write('subl1e = %f\n'%(subl1e[-1]))
    file.write('subl2e = %f\n'%(subl2e[-1]))
    file.write('alpha*subl2e = %f\n'%(alpha*subl2e[-1]))
    file.write('loss = %f\n'%(subl1e[-1]+alpha*subl2e[-1]))
    file.close()


##################
# SEMI SUPERVISED 
##################

import copy
import os
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.metric import dice_coeff
import matplotlib.pyplot as plt
import numpy as np

def update_ema_variables(ema_model, model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def criterion_consistency(p1, p2):
    return torch.mean((p1 - p2) ** 2)

def plot_sample(image, label, pred_student, step, writer=None, save_dir=None, pred_teacher=None):
    image = image.detach().cpu().squeeze().numpy()
    label = label.detach().cpu().squeeze().numpy()
    pred_student = pred_student.detach().cpu().squeeze().numpy()

    fig_cols = 4 if pred_teacher is not None else 3
    fig, axs = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4))

    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("T2 Image")

    axs[1].imshow(label, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(pred_student, cmap="gray")
    axs[2].set_title("Student Prediction")

    if pred_teacher is not None:
        pred_teacher = pred_teacher.detach().cpu().squeeze().numpy()
        axs[3].imshow(pred_teacher, cmap="gray")
        axs[3].set_title("Teacher Prediction")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()

    if save_dir:
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        save_path = os.path.join(save_dir, "plots", f"step_{step:05d}.png")
        plt.savefig(save_path, bbox_inches='tight')

    if writer:
        import io
        import PIL.Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = PIL.Image.open(buf)
        img = np.array(img).transpose(2, 0, 1)[:3]
        writer.add_image("sample/image-label-pred", img, step)
        buf.close()

    plt.close()

def eval_net(net, loader, device):
    net.eval()
    tot = 0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            imgs, masks = batch[0].to(device), batch[1].to(device)
            preds = net(imgs)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            tot += dice_coeff(preds, masks).item()
            n_batches += 1
    return tot / n_batches if n_batches > 0 else 0

def dice_history(epochs, train_dices, output):
    plt.plot(range(len(train_dices)), train_dices)
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.grid()
    plt.savefig(os.path.join(output, 'dice_train_curve.png'))
    plt.close()
    np.save(os.path.join(output, 'dice_train.npy'), np.array(train_dices))




def launch_training(model, train_loader, criterion, optimizer, epochs, save_dir, semi_weight=1.0, enable_plot=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_net = model.to(device)
    teacher_net = copy.deepcopy(model).to(device)
    teacher_net.load_state_dict(model.state_dict())
    teacher_net.eval()

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    global_step = 0
    ema_decay = 0.99
    train_dices = []
    best_dice = -1

    for epoch in range(epochs):
        student_net.train()
        running_loss = 0.0
        epoch_dice_total = 0.0
        epoch_dice_count = 0

        for step, (images, masks) in enumerate(train_loader):
            images = torch.stack(images).to(device)
            preds_student = student_net(images)
            with torch.no_grad():
                preds_teacher = teacher_net(images)

            supervised_loss = 0
            unsupervised_loss = 0
            num_supervised = 0
            num_unsupervised = 0

            for i, mask in enumerate(masks):
                pred = preds_student[i:i+1]
                if mask is not None:
                    mask = mask.unsqueeze(0).to(device)
                    loss_seg = criterion(pred, mask)
                    pred_sigmoid = torch.sigmoid(pred)
                    dice_numerator = 2 * (pred_sigmoid * mask).sum()
                    dice_denominator = (pred_sigmoid + mask).sum()
                    loss_seg_dice = 1 - (dice_numerator + 1e-5) / (dice_denominator + 1e-5)
                    supervised_loss += loss_seg + loss_seg_dice
                    num_supervised += 1
                    epoch_dice_total += dice_coeff((pred_sigmoid > 0.5).float(), mask).item()
                    epoch_dice_count += 1
                else:
                    unsupervised_loss += criterion_consistency(
                        torch.sigmoid(pred), torch.sigmoid(preds_teacher[i:i+1]))
                    num_unsupervised += 1

            if num_supervised > 0:
                supervised_loss /= num_supervised
            if num_unsupervised > 0:
                unsupervised_loss /= num_unsupervised

            total_loss = supervised_loss + semi_weight * unsupervised_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema_variables(teacher_net, student_net, ema_decay, global_step)
            global_step += 1
            running_loss += total_loss.item()

            writer.add_scalar('loss/supervised_total', supervised_loss.item() if num_supervised else 0, global_step)
            writer.add_scalar('loss/unsupervised', unsupervised_loss.item() if num_unsupervised else 0, global_step)
            writer.add_scalar('loss/total', total_loss.item(), global_step)

            # PLots désactivables
            if enable_plot and global_step % 20 == 0:
                for i, mask in enumerate(masks):
                    if mask is not None:
                        img = images[i, 0]
                        lbl = mask[0]
                        pred_student = (torch.sigmoid(preds_student[i, 0]) > 0.5).float()
                        pred_teacher = (torch.sigmoid(preds_teacher[i, 0]) > 0.5).float()
                        plot_sample(img, lbl, pred_student, global_step, writer, save_dir, pred_teacher)
                        break


        epoch_dice = epoch_dice_total / epoch_dice_count if epoch_dice_count > 0 else 0
        train_dices.append(epoch_dice)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logging.info(f"[Epoch {epoch+1}/{epochs}] Loss: {running_loss:.2f} | Dice: {epoch_dice:.4f} ({epoch_dice_count} slices)")

        writer.add_scalar('metrics/dice_epoch', epoch_dice, epoch)

        if epoch_dice > best_dice:
            torch.save(student_net.state_dict(), os.path.join(save_dir, "best_model.pth"))
            best_dice = epoch_dice

        if (epoch + 1) % 100 == 0:
            torch.save(student_net.state_dict(), os.path.join(save_dir, f"student_epoch_{epoch+1}.pth"))
            torch.save(teacher_net.state_dict(), os.path.join(save_dir, f"teacher_epoch_{epoch+1}.pth"))

    dice_history(epochs, train_dices, save_dir)
    writer.close()
    logging.info("Training finished.")



###############################






def launch_training_AE(epochs, net, AE, train_loader, val_loader, n_train, device, net_id, criterion, criterion_reg, alpha, optimizer, output):
    alpha /= 1000.
    train_dices, val_dices = [], []
    best_val_dice = -1
    subl1e, subl2e = [], [] # store averaged sub-losses for each epoch
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        subl1b, subl2b = [], [] # store sub-losses for each batch
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, masks = batch[0], batch[1]
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)
                preds = net(imgs)
                # ======
                latgt, latpr = AE(masks)[1], AE(preds)[1]
                latgt, latpr = torch.torch.flatten(latgt, start_dim=1), torch.torch.flatten(latpr, start_dim=1)
                reg = criterion_reg(latgt,latpr,Variable(torch.ones(latgt.size(0))).cuda())
                # ======
                loss = criterion(preds, masks)
                subloss1, subloss2 = 0.5*loss, 0.5*reg
                subl1b.append(subloss1.detach().cpu())
                subl2b.append(subloss2.detach().cpu())
                loss = subloss1 + alpha*subloss2
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch) ': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
        subl1e.append(np.mean(np.array(subl1b)))
        subl2e.append(np.mean(np.array(subl2b)))
        display_subloss_values(output, epoch, alpha, subl1e, subl2e)
        train_dices.append(eval_net(net, train_loader, device))
        val_dices.append(eval_net(net, val_loader, device))
        write_metric_values(net, output, epoch, train_dices, val_dices)
        if epoch>0:
            if val_dices[-1]>best_val_dice:
                os.remove(output+'epoch.pth')
                torch.save(net.state_dict(), output+'epoch.pth')
                best_val_dice = val_dices[-1]
                logging.info(f'checkpoint {epoch + 1} saved !')
        else:
            torch.save(net.state_dict(), output+'epoch.pth') 
    return train_dices, val_dices, subl1e, subl2e
    
def train_AE(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output):
    train_dices, val_dices = [], []
    best_val_dice = -1
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                masks = batch[1]
                masks = masks.to(device=device, dtype=torch.float32)
                preds, _ = net(masks)
                loss = criterion(preds, masks)                
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(masks.shape[0])
        train_dices.append(eval_net_AE(net, train_loader, device))
        val_dices.append(eval_net_AE(net, val_loader, device))
        write_metric_values(net, output, epoch, train_dices, val_dices)
        if epoch>0:
            if val_dices[-1]>best_val_dice:
                os.remove(output+'epoch.pth')
                torch.save(net.state_dict(), output+'epoch.pth')
                best_val_dice = val_dices[-1]
                logging.info(f'checkpoint {epoch + 1} saved !')
        else:
            torch.save(net.state_dict(), output+'epoch.pth') 
    return train_dices, val_dices


    
def launch_training_MT(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output, gamma):
    train_dices, val_dices = [], []
    best_val_dice = -1
    gamma = gamma/10.
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, masks_LK, masks_RK = batch[0], batch[1], batch[2]
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks_LK = masks_LK.to(device=device, dtype=mask_type)
                masks_RK = masks_RK.to(device=device, dtype=mask_type)
                preds_LK, preds_RK = net(imgs)
                loss_LK = criterion(preds_LK, masks_LK)
                loss_RK = criterion(preds_RK, masks_RK)
                loss = gamma*loss_LK+(1-gamma)*loss_RK
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
        train_dices.append(eval_net_MT(net, train_loader, device))
        val_dices.append(eval_net_MT(net, val_loader, device))
        write_metric_values(net, output, epoch, train_dices, val_dices)
        if epoch>0:
            if val_dices[-1]>best_val_dice:
                os.remove(output+'epoch.pth')
                torch.save(net.state_dict(), output+'epoch.pth')
                best_val_dice = val_dices[-1]
                logging.info(f'checkpoint {epoch + 1} saved !')
        else:
            torch.save(net.state_dict(), output+'epoch.pth') 
    return train_dices, val_dices

def launch_training_3O(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output):
    train_dices, val_dices = [], []
    best_val_dice = -1
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                if len(batch) == 3:
                    imgs, masks_LK, masks_RK = batch[0], batch[1], batch[2]
                    LVhere = False
                elif len(batch) == 4:
                    imgs, masks_LK, masks_RK, masks_LV = batch[0], batch[1], batch[2], batch[3]
                    LVhere = True
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks_LK = masks_LK.to(device=device, dtype=mask_type)
                masks_RK = masks_RK.to(device=device, dtype=mask_type)
                if LVhere:
                    mask_LV = masks_LV.to(device=device, dtype=mask_type)
                preds_LK, preds_RK, preds_LV = net(imgs)
                if LVhere:
                    loss_LK = criterion(preds_LK, masks_LK)
                    loss_RK = criterion(preds_RK, masks_RK)
                    loss_LV = criterion(preds_LV, mask_LV)
                    loss = (loss_LK + loss_RK + loss_LV)/3.
                else:
                    loss_LK = criterion(preds_LK, masks_LK)
                    loss_RK = criterion(preds_RK, masks_RK)
                    loss = (loss_LK + loss_RK)/2.                    
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
        train_dices.append(eval_net_3O(net, train_loader, device))
        val_dices.append(eval_net_3O(net, val_loader, device))
        write_metric_values(net, output, epoch, train_dices, val_dices)
        if epoch>0:
            if val_dices[-1]>best_val_dice:
                os.remove(output+'epoch.pth')
                torch.save(net.state_dict(), output+'epoch.pth')
                best_val_dice = val_dices[-1]
                logging.info(f'checkpoint {epoch + 1} saved !')
        else:
            torch.save(net.state_dict(), output+'epoch.pth') 
    return train_dices, val_dices

def launch_training_MT_AE(epochs, net, AELK, AERK, train_loader, val_loader, n_train, device, net_id, criterion, criterion_reg, alpha, optimizer, output):
    alpha /= 1000.
    train_dices, val_dices = [], []
    best_val_dice = -1
    subl1e, subl2e = [], [] # store averaged sub-losses for each epoch
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        subl1b, subl2b = [], [] # store sub-losses for each batch
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, masks_LK, masks_RK = batch[0], batch[1], batch[2]
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks_LK = masks_LK.to(device=device, dtype=mask_type)
                masks_RK = masks_RK.to(device=device, dtype=mask_type)
                preds_LK, preds_RK = net(imgs)
                # ======
                latLKgt, latLKpr = AELK(masks_LK)[1], AELK(preds_LK)[1]
                latRKgt, latRKpr = AERK(masks_RK)[1], AERK(preds_RK)[1]
                latLKgt, latLKpr = torch.torch.flatten(latLKgt, start_dim=1), torch.torch.flatten(latLKpr, start_dim=1)
                latRKgt, latRKpr = torch.torch.flatten(latRKgt, start_dim=1), torch.torch.flatten(latRKpr, start_dim=1)
                regLK = criterion_reg(latLKgt,latLKpr,Variable(torch.ones(latLKgt.size(0))).cuda())
                regRK = criterion_reg(latRKgt,latRKpr,Variable(torch.ones(latRKgt.size(0))).cuda())
                # ======
                loss_LK, loss_RK = criterion(preds_LK, masks_LK), criterion(preds_RK, masks_RK)
                subloss1, subloss2 = 0.5*(loss_LK + loss_RK), 0.5*(regLK + regRK)
                subl1b.append(subloss1.detach().cpu())
                subl2b.append(subloss2.detach().cpu())
                loss = subloss1 + alpha*subloss2
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch) ': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
        subl1e.append(np.mean(np.array(subl1b)))
        subl2e.append(np.mean(np.array(subl2b)))
        display_subloss_values(output, epoch, alpha, subl1e, subl2e)
        train_dices.append(eval_net_MT(net, train_loader, device))
        val_dices.append(eval_net_MT(net, val_loader, device))
        write_metric_values(net, output, epoch, train_dices, val_dices)
        if epoch>0:
            if val_dices[-1]>best_val_dice:
                os.remove(output+'epoch.pth')
                torch.save(net.state_dict(), output+'epoch.pth')
                best_val_dice = val_dices[-1]
                logging.info(f'checkpoint {epoch + 1} saved !')
        else:
            torch.save(net.state_dict(), output+'epoch.pth') 
    return train_dices, val_dices, subl1e, subl2e

"""
def launch_training_MT_contrastive(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output, clfTask=False, weightmayo=1.):
    loss_mayo = SupConLoss()
    train_dices, val_dices = [], []
    best_val_dice = -1
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                if clfTask:
                    imgs, masks_LK, masks_RK, mayos = batch[0], batch[1], batch[2], batch[3]
                else:
                    imgs, masks_LK, masks_RK = batch[0], batch[1], batch[2]                 
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks_LK = masks_LK.to(device=device, dtype=mask_type)
                masks_RK = masks_RK.to(device=device, dtype=mask_type)
                # import numpy as np
                # bsz = mayos.shape[0]
                # print(np.unique(mayos),bsz)
                if clfTask:
                    mayos = mayos.to(device=device, dtype=torch.float32)
                if clfTask:
                    preds_LK, preds_RK, x5 = net(imgs)
                    x5 = torch.unsqueeze(x5, dim=1)
                    print('coucou',x5.size())
                else:
                    preds_LK, preds_RK = net(imgs)                    
                loss_LK = criterion(preds_LK, masks_LK)
                loss_RK = criterion(preds_RK, masks_RK)
                if clfTask:
                    print('to do')
                    # loss_mayo = torch.nn.L1Loss()(preds_mayos,mayos)
                    # f1, f2 = torch.split(x5, [bsz, bsz], dim=0)
                    # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    # l_mayo = loss_mayo(features=features, labels=mayos)
                    # print(l_mayo.cpu())
                    # weightmayo /= 100.
                    # loss = (loss_LK + loss_RK + weightmayo*l_mayo)/(2+weightmayo)
                else:
                    loss = (loss_LK + loss_RK)/2.
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
        train_dices.append(eval_net_MT(net, train_loader, device, clfTask))
        val_dices.append(eval_net_MT(net, val_loader, device, clfTask))
        write_metric_values(net, output, epoch, train_dices, val_dices)
        if epoch>0:
            if val_dices[-1]>best_val_dice:
                os.remove(output+'epoch.pth')
                torch.save(net.state_dict(), output+'epoch.pth')
                best_val_dice = val_dices[-1]
                logging.info(f'checkpoint {epoch + 1} saved !')
        else:
            torch.save(net.state_dict(), output+'epoch.pth')

    return train_dices, val_dices
"""
    
def eval_net_MT(net, loader, device):
    ''' evaluation with dice coefficient '''
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    tot = 0
    for batch in loader:
        imgs, masks_LK, masks_RK = batch[0], batch[1], batch[2]
        imgs = imgs.to(device=device, dtype=torch.float32)
        masks_LK = masks_LK.to(device=device, dtype=mask_type)
        masks_RK = masks_RK.to(device=device, dtype=mask_type)
        with torch.no_grad():
            preds_LK, preds_RK = net(imgs)
        preds_LK = torch.sigmoid(preds_LK)            
        preds_LK = (preds_LK > 0.5).float()
        preds_RK = torch.sigmoid(preds_RK)            
        preds_RK = (preds_RK > 0.5).float()       
        tot += dice_coeff(preds_LK, masks_LK).item()
        tot += dice_coeff(preds_RK, masks_RK).item()
    return tot/(2*n_val)

def eval_net_3O(net, loader, device):
    ''' evaluation with dice coefficient '''
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    tot = 0
    for batch in loader:
        imgs, masks_LK, masks_RK = batch[0], batch[1], batch[2]
        imgs = imgs.to(device=device, dtype=torch.float32)
        masks_LK = masks_LK.to(device=device, dtype=mask_type)
        masks_RK = masks_RK.to(device=device, dtype=mask_type)
        with torch.no_grad():
            preds_LK, preds_RK, _ = net(imgs)
        preds_LK = torch.sigmoid(preds_LK)            
        preds_LK = (preds_LK > 0.5).float()
        preds_RK = torch.sigmoid(preds_RK)            
        preds_RK = (preds_RK > 0.5).float()       
        tot += dice_coeff(preds_LK, masks_LK).item()
        tot += dice_coeff(preds_RK, masks_RK).item()
    return tot/(2*n_val)
            
def dice_history(epochs, train_dices, val_dices, output):
    ''' display and save dices after training '''
    plt.plot(range(epochs), train_dices)
    plt.plot(range(epochs), val_dices)
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.xlim([0,epochs-1]) 
    plt.ylim([0,1]) # plt.ylim([0.9,1]) 
    plt.grid()
    plt.savefig(output+'dices.png')
    plt.close()
    np.save(output+'train_dices.npy', np.array(train_dices))
    np.save(output+'val_dices.npy', np.array(val_dices))
    
def subloss_history(epochs, subl1e, subl2e, alpha, output):
    subl1e, subl2e = np.array(subl1e), np.array(subl2e)
    plt.plot(range(epochs), subl1e+alpha*subl2e)
    plt.plot(range(epochs), subl1e)
    plt.plot(range(epochs), subl2e)
    plt.legend(['loss', 'subl1', 'subl2'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim([0,epochs-1]) 
    # plt.ylim([0,0.2]) 
    plt.grid()
    plt.savefig(output+'sublosses.png')
    plt.close()
    np.save(output+'subl1.npy', np.array(subl1e))
    np.save(output+'subl2.npy', np.array(subl2e))
    
def cross_entropy_history(epochs, train_cross_entropy, val_cross_entropy, output):
    ''' display and save dices after training '''
    plt.plot(range(epochs), train_cross_entropy)
    plt.plot(range(epochs), val_cross_entropy)
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('cross-entropy')
    plt.xlim([0,epochs-1]) 
    # plt.ylim([0,1]) 
    plt.grid()
    plt.savefig(output+'cross-entropy.png')
    plt.close()
    np.save(output+'train_cross_entropy.npy', np.array(train_cross_entropy))
    np.save(output+'val_cross_entropy.npy', np.array(val_cross_entropy))
    
def eval_net(net, loader, device):
    ''' evaluation with dice coefficient '''
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    tot = 0
    for batch in loader:
        imgs, masks = batch[0], batch[1]
        imgs = imgs.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=mask_type)
        with torch.no_grad():
            preds = net(imgs)
        if net.n_classes > 1:
            tot += nn.functional.cross_entropy(preds,torch.squeeze(masks,1))
        else:
            preds = torch.sigmoid(preds)            
            preds = (preds > 0.5).float()
            tot += dice_coeff(preds, masks).item()
    return tot / n_val

def eval_net_AE(net, loader, device):
    ''' evaluation with dice coefficient '''
    net.eval()
    n_val = len(loader)
    tot = 0
    for batch in loader:
        masks = batch[1]
        masks = masks.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            preds, _ = net(masks)
        preds = torch.sigmoid(preds)            
        preds = (preds > 0.5).float()
        tot += dice_coeff(preds, masks).item()
    return tot / n_val

def supervision_loss(attention, preds, masks, criterion):
    if attention:
        loss1 = criterion(preds[0], masks)
        loss2 = criterion(preds[1], masks)
        loss3 = criterion(preds[2], masks)
        loss4 = criterion(preds[3], masks)
        loss5 = criterion(preds[4], masks)
        loss6 = criterion(preds[5], masks)
        loss7 = criterion(preds[6], masks)
        loss8 = criterion(preds[7], masks)
        loss9 = criterion(preds[8], masks)
        loss = loss1+0.8*loss2+0.7*loss3+0.6*loss4+0.5*loss5+0.8*loss6+0.7*loss7+0.6*loss8+0.5*loss9
        loss /= 6.2
    else:
        loss1 = criterion(preds[0], masks)
        loss2 = criterion(preds[1], masks)
        loss3 = criterion(preds[2], masks)
        loss4 = criterion(preds[3], masks)
        loss5 = criterion(preds[4], masks)
        loss = loss1+0.8*loss2+0.7*loss3+0.6*loss4+0.5*loss5    
        loss /= 3.6        
    return loss