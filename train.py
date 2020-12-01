from torch import nn
from torch.autograd import Variable
from random import randrange
from models.Discriminator import *
from test import main as test_main
from utils.utils import *
from dataloader import *
from pathlib import Path
import torch.nn.functional as F
from argparse import ArgumentParser
from models.Unet import *
import random

# import

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config_train.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    n_channel = config['n_channel']
    normal_class = config["normal_class"]

    checkpoint_path = "outputs/{}/{}/checkpoints/".format(config['dataset_name'], normal_class)
    train_output_path = "outputs/{}/{}/train_outputs/".format(config['dataset_name'], normal_class)

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(train_output_path).mkdir(parents=True, exist_ok=True)

    epsilon = float(config['eps'])
    alpha = float(config['alpha'])

    train_dataloader, _, _ = load_data(config)

    unet = UNet(n_channel, n_channel, config['base_channel']).cuda()
    discriminator = NetD(config['image_size'], n_channel, config['n_extra_layers']).cuda()
    discriminator.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer_u = torch.optim.Adam(
        unet.parameters(), lr=config['lr_u'], weight_decay=float(config['weight_decay']))

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_u, factor=config['factor'], patience=config['patience'], mode='min', verbose=True)

    ae_loss_list = []

    l_adv = l2_loss
    l_bce = nn.BCELoss()

    permutation_list = get_all_permutations()

    # num_epochs = config['num_epochs']
    num_epochs = 15
    epoch_loss_dict = dict()
    unet.train()
    discriminator.train()
    for _ in range(20):
        rot_weight = 10 ** (random.uniform(-2, 1))
        print(rot_weight)
        for epoch in range(num_epochs + 1):
            epoch_ae_loss = 0
            epoch_total_loss = 0
            epoch_disc_loss = 0

            for data in train_dataloader:
                rand_number = randrange(4)
                img = data[0]
                orig_img = img

                partitioned_img, base = split_tensor(img, tile_size=img.size(2) // 2, offset=img.size(2) // 2)
                perm = get_random_permutation()

                extended_perm = perm * img.size(1)
                if img.size(1) == 3:
                    offset = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
                    final_perm = offset + extended_perm[:, None]
                    final_perm = final_perm.view(-1)
                else:
                    final_perm = extended_perm

                permuted_img = partitioned_img[:, final_perm, :, :]

                if img.size(1) == 3:
                    avg = permuted_img[:, rand_number * 3, :, :] + permuted_img[:, rand_number * 3 + 1, :, :] + \
                          permuted_img[:, rand_number * 3 + 2, :, :]

                    avg /= 3
                    permuted_img[:, rand_number * 3, :, :] = avg
                    permuted_img[:, rand_number * 3 + 1, :, :] = avg
                    permuted_img[:, rand_number * 3 + 2, :, :] = avg
                else:
                    permuted_img[:, rand_number, :, :] *= 0

                target = orig_img
                permuted_img = rebuild_tensor(permuted_img, base, tile_size=img.size(2) // 2, offset=img.size(2) // 2)

                permuted_img = fgsm_attack(permuted_img, unet, eps=epsilon, alpha=alpha)

                img = Variable(permuted_img).cuda()
                target = Variable(target).cuda()

                # ===================forward=====================

                # Forward Unet
                output = unet(img)

                # Forward Discriminator
                x = output.detach()
                x_90 = x.transpose(2, 3)
                x_180 = x.flip(2, 3)
                x_270 = x.transpose(2, 3).flip(2, 3)
                generated_data = torch.cat((x, x_90, x_180, x_270), 0)

                x = target
                x_90 = x.transpose(2, 3)
                x_180 = x.flip(2, 3)
                x_270 = x.transpose(2, 3).flip(2, 3)
                real_data = torch.cat((x, x_90, x_180, x_270), 0)

                pred_real, classification_real, rot_probs_real, feat_real = discriminator(real_data)
                pred_fake, classification_fake, rot_probs_fake, feat_fake = discriminator(generated_data.detach())

                # ===================backward====================

                # Backward Unet
                optimizer_u.zero_grad()
                err_g_adv = l_adv(discriminator(real_data)[3], discriminator(generated_data)[3])
                AE_loss = criterion(output, target)
                loss = config['adv_coeff'] * err_g_adv + AE_loss

                epoch_total_loss += loss.item()
                epoch_ae_loss += AE_loss.item()
                loss.backward()
                optimizer_u.step()

                # Backward Discriminator
                real_label = torch.ones(size=(img.shape[0] * 4,), dtype=torch.float32).cuda()
                fake_label = torch.zeros(size=(img.shape[0] * 4,), dtype=torch.float32).cuda()

                optimizer_d.zero_grad()
                err_d_real = l_bce(pred_real, real_label)
                err_d_fake = l_bce(pred_fake, fake_label)

                batch_size = img.shape[0]
                rot_labels = torch.zeros(4 * batch_size).cuda()
                for i in range(4 * batch_size):
                    if i < batch_size:
                        rot_labels[i] = 0
                    elif i < 2 * batch_size:
                        rot_labels[i] = 1
                    elif i < 3 * batch_size:
                        rot_labels[i] = 2
                    else:
                        rot_labels[i] = 3
                rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()
                d_real_class_loss = torch.sum(F.binary_cross_entropy_with_logits(
                    input=classification_real,
                    target=rot_labels))
                # err_d_rot = d_real_class_loss * config['rot_loss_weight']
                err_d_rot = d_real_class_loss * rot_weight
                err_d = (err_d_real + err_d_fake) * 0.5 + err_d_rot
                err_d.backward()
                optimizer_d.step()
                epoch_disc_loss += err_d

            # ===================log========================
            ae_loss_list.append(epoch_ae_loss)
            scheduler.step(epoch_ae_loss)

            # print('epoch [{}/{}], epoch_total_loss:{:.4f}, epoch_ae_loss:{:.4f}, err_adv:{:.4f}, err_disc:{:.4f}'
            #       .format(epoch + 1, num_epochs, epoch_total_loss, epoch_ae_loss, err_g_adv, epoch_disc_loss))

            with open(checkpoint_path + 'log_{}.txt'.format(normal_class), "a") as log_file:
                log_file.write('\n epoch [{}/{}], loss:{:.4f}, epoch_ae_loss:{:.4f}, err_adv:{:.4f}'
                               .format(epoch + 1, num_epochs, epoch_total_loss, epoch_ae_loss, err_g_adv))

            if epoch % 20 == 11:
                test_main(epoch - 1)

            if epoch % 10 == 0:
                show_process_for_trainortest(img, output, orig_img, train_output_path + str(epoch))
                epoch_loss_dict[epoch] = epoch_total_loss

                torch.save(unet.state_dict(), checkpoint_path + '{}.pth'.format(str(epoch)))
                torch.save(discriminator.state_dict(), checkpoint_path + 'netd_{}.pth'.format(str(epoch)))

                torch.save(optimizer_u.state_dict(), checkpoint_path + 'opt_{}.pth'.format(str(epoch)))
                torch.save(optimizer_d.state_dict(), checkpoint_path + 'optd_{}.pth'.format(str(epoch)))

                torch.save(scheduler.state_dict(), checkpoint_path + 'scheduler_{}.pth'.format(str(epoch)))


if __name__ == '__main__':
    main()
