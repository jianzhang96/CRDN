import torch
from data_loader import MVTecTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork
from model_unet_bica import DiscriminativeSubNetwork_BiCA
from loss import FocalLoss, SSIM
import os
import time, datetime

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        print(obj_name)
        run_name = 'CRDN_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))
        '''
        size    | base_width | base_channels
        tiny    | 16         | 8
        small   | 32         | 16
        base    | 64         | 32
        large   | 128        | 64
        By default, we choose CRDN-Base model
        '''
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=64)
        model.cuda()
        model.apply(weights_init)

        model2 = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=64)
        model2.cuda()
        model2.apply(weights_init)

        model_seg = DiscriminativeSubNetwork_BiCA(in_channels=6, out_channels=2, base_channels=32)
        model_seg.cuda()
        model_seg.apply(weights_init)


        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr},
                                      {"params": model2.parameters(), "lr": args.lr},])  ##

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])

        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)

        start_time = time.time()##
        iter_pb = len(dataset)/args.bs
        n_iter = 0
        for epoch in range(args.epochs):
            # print("Epoch: "+str(epoch))
            loss_ep = 0
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()

                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                ##
                gray_rec2 = model2(gray_rec)
                joined_in2 = torch.cat((gray_rec2, aug_gray_batch), dim=1)

                out_mask, out_mask2 = model_seg(joined_in,joined_in2)

                out_mask_sm = torch.softmax(out_mask, dim=1)
                out_mask_sm2 = torch.softmax(out_mask2, dim=1)
                # loss2
                l2_loss2 = loss_l2(gray_rec2, gray_batch)
                ssim_loss2 = loss_ssim(gray_rec2, gray_batch)

                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)

                if obj_name in ['pill', 'capsule']:
                    out_mask_avg = (out_mask_sm + out_mask_sm2)/2
                    segment_loss = loss_focal(out_mask_avg, anomaly_mask)
                else:
                    segment_loss = loss_focal(out_mask_sm, anomaly_mask) + loss_focal(out_mask_sm2, anomaly_mask)

                loss = l2_loss + ssim_loss + segment_loss + \
                       l2_loss2 + ssim_loss2

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                    visualizer.plot_loss(l2_loss2, n_iter, loss_name='l2_loss2') ##
                    visualizer.plot_loss(ssim_loss2, n_iter, loss_name='ssim_loss2')
                    # visualizer.plot_loss(segment_loss2, n_iter, loss_name='segment_loss2')
                if args.visualize and n_iter % 400 == 0:
                    t_mask = out_mask_sm[:, 1:, :, :]
                    visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                    visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                    visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                    visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')
                    ##
                    t_mask2 = out_mask_sm2[:, 1:, :, :]
                    visualizer.visualize_image_batch(gray_rec2, n_iter, image_name='batch_recon_out2')
                    visualizer.visualize_image_batch(t_mask2, n_iter, image_name='mask_out2')


                n_iter +=1
                loss_ep += loss

            scheduler.step()

            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))
            torch.save(model2.state_dict(), os.path.join(args.checkpoint_path, run_name+"_2.pckl")) ##


            eta_seconds = ((time.time() - start_time) / (epoch+1e-5)) * (args.epochs - epoch) ###
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print("Epoch: {:d}/{:d}  || Loss: {:.4f} || Cost: {} || Est: {}".format(
                            epoch, args.epochs, loss/iter_pb,
                            str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args.obj_id) == -1:
        obj_list = [
                     'toothbrush',
                     'carpet',
                     'zipper',
                     'transistor',
                     'leather',
                     'capsule',
                     'pill',
                     'grid',
                     'screw',
                     'bottle',
                     'tile',
                     'hazelnut',
                     'cable',
                     'wood',
                     'metal_nut',
                     ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

