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
import logging
from torch.utils.tensorboard import SummaryWriter


ema_tracking_history = {
    "ema": [],
    "student": [],
    "diff": [],
    "steps": []
}

def disable_bn_tracking(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = False

def update_ema_variables(student_model, teacher_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data.detach(), alpha=1 - alpha)

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

def log_ema_tracking_matplotlib(student_net, teacher_net, step, save_dir):
    with torch.no_grad():
        for (name_s, param_s), (name_t, param_t) in zip(student_net.named_parameters(), teacher_net.named_parameters()):
            if "weight" in name_s and param_s.numel() > 10:
                student_val = param_s.mean().item()
                teacher_val = param_t.mean().item()
                diff_val = (param_s - param_t).abs().mean().item()

                ema_tracking_history["student"].append(student_val)
                ema_tracking_history["ema"].append(teacher_val)
                ema_tracking_history["diff"].append(diff_val)
                ema_tracking_history["steps"].append(step)

                if len(ema_tracking_history["steps"]) % 5 == 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(ema_tracking_history["steps"], ema_tracking_history["ema"], label="EMA Param", color="orange")
                    plt.plot(ema_tracking_history["steps"], ema_tracking_history["student"], label="Student Param", linestyle="--", color="blue")
                    plt.plot(ema_tracking_history["steps"], ema_tracking_history["diff"], label="Diff", linestyle=":", color="green")
                    plt.xlabel("Iteration")
                    plt.ylabel("Mean Value")
                    plt.title("EMA Update Tracking")
                    plt.legend()
                    plt.grid(True)
                    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
                    plt.savefig(os.path.join(save_dir, "plots", "ema_tracking_plot.png"))
                    plt.close()
                break


def plot_metric_curve(values, ylabel, save_path):
    plt.plot(range(len(values)), values)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def save_all_metric_curves(train_dices, val_dices_student, val_dices_teacher,
                           bce_losses, dice_losses, mse_losses, total_losses,
                           lambda_consistencies,  
                           save_dir):
    metrics = {
        'dice_train': (train_dices, 'Dice (Train)', 'dice_train_curve.png'),
        'dice_val_student': (val_dices_student, 'Dice (Val - Student)', 'dice_val_student_curve.png'),
        'dice_val_teacher': (val_dices_teacher, 'Dice (Val - Teacher)', 'dice_val_teacher_curve.png'),
        'bce_loss': (bce_losses, 'BCE Loss', 'loss_bce_curve.png'),
        'dice_loss': (dice_losses, 'Dice Loss', 'loss_dice_curve.png'),
        'mse_loss': (mse_losses, 'MSE Loss', 'loss_mse_curve.png'),
        'total_loss': (total_losses, 'Total Loss', 'loss_total_curve.png'),
        'lambda_consistency': (lambda_consistencies, 'Lambda Consistency', 'lambda_consistency_curve.png') 
    }
    for key, (values, ylabel, filename) in metrics.items():
        plot_metric_curve(values, ylabel, os.path.join(save_dir, filename))
        np.save(os.path.join(save_dir, f"{key}.npy"), np.array(values))



def sigmoid_rampup(current, rampup_length):# Consistency ramp-up from https://arxiv.org/abs/1610.02242
        
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


def launch_training(model, train_loader, val_loader, criterion, epochs, save_dir, lambda_consistency=0, enable_plot=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_net = model.to(device)
    teacher_net = copy.deepcopy(model).to(device)
    teacher_net.load_state_dict(model.state_dict())
    for param in teacher_net.parameters():
        param.detach_()

    student_net.train()
    optimizer = torch.optim.Adam(student_net.parameters(), lr=1e-3)

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    global_step = 0
    train_dices = []
    val_dices_student = []
    val_dices_teacher = []
    total_losses = []
    bce_losses = []
    dice_losses = []
    mse_losses = []
    lambda_consistencies = []
    best_dice_student = -1
    best_dice_teacher = -1
    rampup_length = 50
    epochs_decay = 250


# Enregistrement poids - optimiseur -scheduler pour relancer train si jamais coupure
    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        teacher_net.load_state_dict(checkpoint['teacher_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_dice_student = checkpoint['best_dice_student']
        best_dice_teacher = checkpoint['best_dice_teacher']
        train_dices = checkpoint['train_dices']
        val_dices_student = checkpoint['val_dices_student']
        val_dices_teacher = checkpoint['val_dices_teacher']
        total_losses = checkpoint['total_losses']
        bce_losses = checkpoint['bce_losses']
        dice_losses = checkpoint['dice_losses']
        mse_losses = checkpoint['mse_losses']
        lambda_consistencies = checkpoint.get('lambda_consistencies', [])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Reprise de l'entraînement à l'époque {start_epoch}")


    for epoch in range(start_epoch, epochs):


        rampup = sigmoid_rampup(epoch, rampup_length)
        current_lambda_consistency = lambda_consistency * rampup
        writer.add_scalar('hyperparam/lambda_consistency', current_lambda_consistency, epoch)
        writer.add_scalar('debug/rampup', rampup, epoch)

        writer.add_scalar('hyperparam/lambda_consistency', current_lambda_consistency, epoch)
        lambda_consistencies.append(current_lambda_consistency)


        current_ema_alpha = 0.99
        student_net.train()


        #params_in_optimizer = list(optimizer.param_groups[0]['params'])
        #params_in_optimizer_ids = [id(p) for p in params_in_optimizer]
        #teacher_params_ids = [id(p) for p in teacher_net.parameters()]
        #student_params_ids = [id(p) for p in student_net.parameters()]

        #print("Teacher params in optimizer:", any(pid in params_in_optimizer_ids for pid in teacher_params_ids))
        #print("Student params in optimizer:", any(pid in params_in_optimizer_ids for pid in student_params_ids))

        #assert not any(pid in params_in_optimizer_ids for pid in teacher_params_ids), \
            #"Teacher parameters are incorrectly in the optimizer!"

        running_loss = 0.0
        epoch_dice_total = 0.0
        epoch_dice_count = 0
        epoch_bce_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_mse_loss = 0.0
        num_bce = 0
        num_dice = 0
        num_mse = 0

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
                    loss_seg = criterion(pred.float(), mask.float())  #BCE

                    pred_sigmoid = torch.sigmoid(pred)
                    dice_numerator = 2 * (pred_sigmoid * mask).sum()
                    dice_denominator = (pred_sigmoid + mask).sum()
                    loss_seg_dice = 1 - (dice_numerator + 1e-5) / (dice_denominator + 1e-5) # 1 - DICE
                    
                    supervised_loss += 0.5 * (loss_seg + loss_seg_dice)
                    epoch_bce_loss += loss_seg.item()
                    epoch_dice_loss += loss_seg_dice.item()
                    num_bce += 1
                    num_dice += 1
                    num_supervised += 1
                    epoch_dice_total += dice_coeff((pred_sigmoid > 0.5).float(), mask).item()
                    epoch_dice_count += 1
                else :    
                    mse = criterion_consistency(torch.sigmoid(pred), torch.sigmoid(preds_teacher[i:i+1]))
                    unsupervised_loss += mse
                    epoch_mse_loss += mse.item()
                    num_unsupervised += 1
                    num_mse += 1

            if num_supervised > 0:
                supervised_loss /= num_supervised
            if num_unsupervised > 0:
                unsupervised_loss /= num_unsupervised

            total_loss = supervised_loss + current_lambda_consistency * unsupervised_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            update_ema_variables(student_net, teacher_net, current_ema_alpha, global_step)

            #if global_step % 50 == 0:
                #log_ema_tracking_matplotlib(student_net, teacher_net, global_step, save_dir)

            global_step += 1
            running_loss += total_loss.item()

            writer.add_scalar('loss/supervised_total', supervised_loss.item() if num_supervised else 0, epoch)
            writer.add_scalar('loss/unsupervised', unsupervised_loss.item() if num_unsupervised else 0, epoch)
            writer.add_scalar('loss/total', total_loss.item(), epoch)

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
        val_dice = eval_net(student_net, val_loader, device)
        teacher_training_mode = teacher_net.training
        val_dice_teacher = eval_net(teacher_net, val_loader, device)
        teacher_net.train(teacher_training_mode)


        train_dices.append(epoch_dice)
        val_dices_student.append(val_dice)
        val_dices_teacher.append(val_dice_teacher)
        total_losses.append(running_loss / len(train_loader))
        bce_losses.append(epoch_bce_loss / num_bce if num_bce > 0 else 0)
        dice_losses.append(epoch_dice_loss / num_dice if num_dice > 0 else 0)
        mse_losses.append(epoch_mse_loss / num_mse if num_mse > 0 else 0)



        
        writer.add_scalar('metrics/dice_train', epoch_dice, epoch)
        writer.add_scalar('metrics/dice_val', val_dice, epoch)
        writer.add_scalar('metrics/dice_val_teacher', val_dice_teacher, epoch)


        save_all_metric_curves(
        train_dices, val_dices_student, val_dices_teacher,
        bce_losses, dice_losses,
        mse_losses, total_losses, lambda_consistencies,
        save_dir
    )


        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            logging.info(f"[Epoch {epoch+1}/{epochs}] Train Dice: {epoch_dice:.4f} | Val Dice Student: {val_dice:.4f} | Val Dice Teacher: {val_dice_teacher:.4f} ")

        
        if val_dice > best_dice_student:
            best_dice_student = val_dice
            student_path = os.path.join(save_dir, "best_model_student.pth")
            torch.save(student_net.state_dict(), student_path)
            logging.info(f"[Student] Nouveau meilleur modèle sauvegardé (Val Dice: {val_dice:.4f})")

        if val_dice_teacher > best_dice_teacher:
            best_dice_teacher = val_dice_teacher
            teacher_path = os.path.join(save_dir, "best_model_teacher.pth")
            torch.save(teacher_net.state_dict(), teacher_path)
            logging.info(f"[Teacher] Nouveau meilleur modèle sauvegardé (Val Dice: {val_dice_teacher:.4f})")


        checkpoint = {
        'epoch': epoch,
        'model_state_dict': student_net.state_dict(),
        'teacher_state_dict': teacher_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice_student': best_dice_student,
        'best_dice_teacher': best_dice_teacher,
        'train_dices': train_dices,
        'val_dices_student': val_dices_student,
        'val_dices_teacher': val_dices_teacher,
        'total_losses': total_losses,
        'bce_losses': bce_losses,
        'dice_losses': dice_losses,
        'mse_losses': mse_losses,
        'lambda_consistencies': lambda_consistencies
    }

        torch.save(checkpoint, os.path.join(save_dir, "checkpoint_latest.pth"))
    



        if epoch > 0 and epoch % epochs_decay == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                logging.info(f"LR diminué à {param_group['lr']}")

    writer.close()
    logging.info("Training finished.")

