import os
from time import time
from sys import stdout

import h5py as h5
import numpy as np
import torch

from lib.networks.utils import AverageMeter
from lib.networks.utils import distChamferCUDA, f_score, pairwise_CD, JSD, COV, MMD, KNN
from lib.metrics.evaluation_metrics import emd_approx, EMD_CD_F1


def evaluate(iterator, model, loss_func, **kwargs):
    train_mode = kwargs.get('train_mode')
    util_mode = kwargs.get('util_mode')
    is_saving = kwargs.get('saving')

    if is_saving:
        # saving generated point clouds, ground-truth point clouds.
        clouds_fname = '{}_{}_{}_{}_clouds_{}.h5'.format(kwargs['model_name'][:-4],
                                                         iterator.dataset.part,
                                                         kwargs['cloud_size'],
                                                         kwargs['sampled_cloud_size'],
                                                         util_mode)
        clouds_fname = os.path.join(kwargs['path2save'], clouds_fname)
        print(clouds_fname)
        clouds_file = h5.File(clouds_fname, 'w')
        sampled_clouds = clouds_file.create_dataset(
            'sampled_clouds',
            shape=(kwargs['N_sets'] * len(iterator.dataset), 3, kwargs['sampled_cloud_size']),
            dtype=np.float32
        )
        gt_clouds = clouds_file.create_dataset(
            'gt_clouds',
            shape=(kwargs['N_sets'] * len(iterator.dataset), 3, kwargs['cloud_size']),
            dtype=np.float32
        )

        if train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
            print('save images')
            image_clouds = clouds_file.create_dataset(
                'image_clouds',
                shape=(kwargs['N_sets'] * len(iterator.dataset), 4, 224, 224),
                dtype=np.float32
            )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    inf_time = AverageMeter()

    if util_mode == 'training':
        LB = AverageMeter()
        PNLL = AverageMeter()
        GNLL = AverageMeter()
        GENT = AverageMeter()

    elif util_mode == 'reconstruction':
        CD = AverageMeter()
        EMD = AverageMeter()
        F1 = AverageMeter()

    model.eval()
    torch.set_grad_enabled(False)

    end = time()

    for i, batch in enumerate(iterator):
        data_time.update(time() - end)

        g_clouds = batch['cloud'].cuda(non_blocking=True)
        p_clouds = batch['eval_cloud'].cuda(non_blocking=True)

        inf_end = time()

        # for test, generate samples
        with torch.no_grad():
            if train_mode == 'p_rnvp_mc_g_rnvp_vae':
                inf_end = time()
                outputs = model(g_clouds, p_clouds, n_sampled_points=kwargs['sampled_cloud_size'])
            elif train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
                images = batch['image'].cuda(non_blocking=True)
                inf_end = time()
                outputs = model(g_clouds, p_clouds, images, n_sampled_points=kwargs['sampled_cloud_size'])

        inf_time.update((time() - inf_end) / g_clouds.shape[0], g_clouds.shape[0])

        if util_mode == 'training':
            loss, pnll, gnll, gent = loss_func(g_clouds, p_clouds, outputs)
            LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])
            PNLL.update(pnll.item(), g_clouds.shape[0])
            GNLL.update(gnll.item(), g_clouds.shape[0])
            GENT.update(gent.item(), g_clouds.shape[0])

        elif util_mode == 'reconstruction':
            r_clouds = outputs['p_prior_samples'][-1] # reconstructed

            if kwargs['unit_scale_evaluation']:
                if kwargs['cloud_scale']:
                    r_clouds *= kwargs['cloud_scale_scale']
                    p_clouds *= kwargs['cloud_scale_scale']

            if kwargs['orig_scale_evaluation']:
                if kwargs['cloud_scale']:
                    r_clouds *= kwargs['cloud_scale_scale']
                    p_clouds *= kwargs['cloud_scale_scale']

                if kwargs['cloud_translate']:
                    shift = torch.from_numpy(np.array(kwargs['cloud_translate_shift']).reshape(1, -1, 1)).cuda()
                    r_clouds += shift
                    p_clouds += shift

                if not kwargs['cloud_rescale2orig']:
                    r_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()
                    p_clouds *= batch['orig_s'].unsqueeze(1).unsqueeze(2).cuda()

                if not kwargs['cloud_recenter2orig']:
                    r_clouds += batch['orig_c'].unsqueeze(2).cuda()
                    p_clouds += batch['orig_c'].unsqueeze(2).cuda()

            if is_saving:
                # saving generated point clouds, ground-truth point clouds and sampled labels.
                sampled_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + r_clouds.shape[0]] = r_clouds.detach().cpu().numpy().astype(np.float32)
                gt_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + p_clouds.shape[0]] = p_clouds.detach().cpu().numpy().astype(np.float32)
                if train_mode == 'p_rnvp_mc_g_rnvp_vae_ic':
                    # if have images, save images
                    image_clouds[kwargs['batch_size'] * i:kwargs['batch_size'] * i + images.shape[0]] = images.cpu().numpy().astype(np.float32)

            # when do svr reconstruction, we compute results over batches, because its data size is too large.
            r_clouds = torch.transpose(r_clouds, 1, 2).contiguous()
            p_clouds = torch.transpose(p_clouds, 1, 2).contiguous()

            if kwargs['cd']:
                dl, dr = distChamferCUDA(r_clouds, p_clouds)
                cd = (dl.mean(1) + dr.mean(1)).mean()
                CD.update(cd.item(), p_clouds.shape[0])
            if kwargs['emd']:
                emd = emd_approx(r_clouds, p_clouds).mean()
                EMD.update(emd.item(), p_clouds.shape[0])
            if kwargs['f1']:
                f1 = f_score(r_clouds, p_clouds).mean()
                F1.update(f1.item(), p_clouds.shape[0])

        batch_time.update(time() - end)
        end = time()

    print('Inference time: {} sec/sample'.format(inf_time.avg))

    if util_mode == 'reconstruction':
        # compute cd, emd adn f1 for reconstruction tasks
        if kwargs['cd']:
            print('CD: {:.6f}'.format(CD.avg))
        if kwargs['emd']:
            print('EMD: {:.6f}'.format(EMD.avg))
        if kwargs['f1']:
            print('F1: {:.6f}'.format(F1.avg))
        # res = [CD.avg, EMD.avg]
        res = [CD.avg, EMD.avg, F1.avg]

    if is_saving:
        clouds_file.close()

    return res