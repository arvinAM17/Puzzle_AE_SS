from torch import nn
from random import randrange
from models.Discriminator import *
from test import *
from dataloader import *

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config_train.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    n_channel = config['n_channel']
    checkpoint_path = config['checkpoint_path']
    normal_class = config["normal_class"]

    train_dataloader, _, _ = load_data(config)

    unet = UNet(n_channel, n_channel).cuda()
    discriminator = NetD().cuda()
    discriminator.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer_u = torch.optim.Adam(
        unet.parameters(), lr=config['lr_u'], weight_decay=config['weight_decay'])

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_u, factor=config['factor'], patience=config['patience'], mode='min', verbose=True)

    ae_loss_list = []

    l_adv = l2_loss
    l_bce = nn.BCELoss()

    permutation_list = get_all_permutations()

    num_epochs = config['num_epochs']
    epoch_loss_dict = dict()
    unet.train()
    discriminator.train()

    for epoch in range(num_epochs):

        epoch_ae_loss = 0
        epoch_total_loss = 0

        for data in train_dataloader:
            rand_number = randrange(4)
            img = data[0]
            orig_img = img

            partitioned_img, base = split_tensor(img, tile_size=img.size(2) // 2, offset=img.size(2) // 2)
            perm = get_random_permutation()
            permuted_img = partitioned_img[:, perm, :, :]
            permuted_img[:, rand_number, :, :] *= 0
            target = orig_img
            permuted_img = rebuild_tensor(permuted_img, base, tile_size=img.size(2) // 2, offset=img.size(2) // 2)

            permuted_img = fgsm_attack(permuted_img, unet)

            img = Variable(permuted_img).cuda()
            target = Variable(target).cuda()

            # ===================forward=====================

            # Forward Unet
            output = unet(img)

            # Forward Discriminator
            pred_real, feat_real = discriminator(target)
            pred_fake, feat_fake = discriminator(output.detach())

            # ===================backward====================

            # Backward Unet
            optimizer_u.zero_grad()
            err_g_adv = l_adv(discriminator(target)[1], discriminator(output)[1])
            AE_loss = criterion(output, target)
            loss = config['adv_coeff'] * err_g_adv + AE_loss

            epoch_total_loss += loss.item()
            epoch_ae_loss += AE_loss.item()
            loss.backward()
            optimizer_u.step()

            # Backward Discriminator
            real_label = torch.ones(size=(img.shape[0],), dtype=torch.float32).cuda()
            fake_label = torch.zeros(size=(img.shape[0],), dtype=torch.float32).cuda()

            optimizer_d.zero_grad()
            err_d_real = l_bce(pred_real, real_label)
            err_d_fake = l_bce(pred_fake, fake_label)

            err_d = (err_d_real + err_d_fake) * 0.5
            err_d.backward()
            optimizer_d.step()

        # ===================log========================
        ae_loss_list.append(epoch_ae_loss)
        scheduler.step(epoch_ae_loss)

        print('epoch [{}/{}], epoch_total_loss:{:.4f}, epoch_ae_loss:{:.4f}, err_adv:{:.4f}'
              .format(epoch + 1, num_epochs, epoch_total_loss, epoch_ae_loss, err_g_adv))

        with open(checkpoint_path + '/log_{}.txt'.format(normal_class), "a") as log_file:
            log_file.write('\n epoch [{}/{}], loss:{:.4f}, epoch_ae_loss:{:.4f}, err_adv:{:.4f}'
                           .format(epoch + 1, num_epochs, epoch_total_loss, epoch_ae_loss, err_g_adv))


        if epoch % 50 == 0:
            show_process_for_trainortest(img, output, orig_img)
            epoch_loss_dict[epoch] = epoch_total_loss

            torch.save(unet.state_dict(), checkpoint_path + '/{}.pth'.format(normal_class))
            torch.save(discriminator.state_dict(), checkpoint_path + '/netd_{}.pth'.format(normal_class))

            torch.save(optimizer_u.state_dict(), checkpoint_path + '/opt_{}.pth'.format(normal_class))
            torch.save(optimizer_d.state_dict(), checkpoint_path + '/optd_{}.pth'.format(normal_class))

            torch.save(scheduler.state_dict(), checkpoint_path + '/scheduler_{}.pth'.format(normal_class))
