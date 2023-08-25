import os
from time import time
from sys import stdout

import torch

from lib.networks.utils import AverageMeter, save_model


def train(iterator, model, model_d1, loss_func, gan_loss_func, optimizer, optimizer_d1, scheduler, epoch, iter, **kwargs):
    num_workers = kwargs.get('num_workers')
    train_mode = kwargs.get('train_mode')
    model_name = os.path.join(kwargs['path2save'], 'models', kwargs.get('model_name'))

    batch_time = AverageMeter()
    data_time = AverageMeter()

    LB = AverageMeter()
    PNLL = AverageMeter()
    GNLL = AverageMeter()
    GENT = AverageMeter()

    model.train()
    torch.set_grad_enabled(True)

    end = time()
    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        data_time.update(time() - end)
        scheduler(optimizer, epoch, iter + i)

        g_clouds = batch['cloud'].cuda(non_blocking=True) # (16, 3, 2500)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True) # (16, 3, 2500)

        if train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
            images = batch['image'].cuda(non_blocking=True) # (16, 4, 224, 224)
            ## g_input and p_input are both from GT mesh, g_input is used as input, p_input is used as GT during optimization
            outputs = model(g_clouds, p_clouds, images)
            r_clouds = outputs['p_prior_samples'][-1]
            if epoch > 5:
                i_img_gt, i_img_pred = model_d1(p_clouds, images, r_clouds)
        loss, pnll, gnll, gent = loss_func(g_clouds, p_clouds, outputs)

        with torch.no_grad():
            if torch.isnan(loss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        PNLL.update(pnll.item(), g_clouds.shape[0])
        GNLL.update(gnll.item(), g_clouds.shape[0])
        GENT.update(gent.item(), g_clouds.shape[0])
        LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_adversarial1 = 0
        if epoch > 5:
            optimizer_d1.zero_grad()
            loss_adversarial1 = gan_loss_func(i_img_gt, i_img_pred, False)
            loss_adversarial1.backward()
            optimizer_d1.step()

        batch_time.update(time() - end)
        if (iter + i + 1) % (num_workers) == 0:
            line = 'Epoch: [{0}][{1}/{2}]'.format(epoch + 1, iter + i + 1, len(iterator))
            line += '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time)
            line += '\tLB {LB.val:.2f} ({LB.avg:.2f})'.format(LB=LB)
            line += '\tPNLL {PNLL.val:.2f} ({PNLL.avg:.2f})'.format(PNLL=PNLL)
            line += '\tGNLL {GNLL.val:.2f} ({GNLL.avg:.2f})'.format(GNLL=GNLL)
            line += '\tGENT {GENT.val:.2f} ({GENT.avg:.2f})'.format(GENT=GENT)
            line += '\t||  GANL_Global {GANL_Global:.4f}'.format(GANL_Global=loss_adversarial1)
            line += '\n'
            stdout.write(line)
            stdout.flush()

        end = time()

        if (iter + i + 1) % (100 * num_workers) == 0:
            save_model({
                'epoch': epoch,
                'iter': iter + i + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, model_name)

    save_model({
        'epoch': epoch + 1,
        'iter': 0,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, model_name)