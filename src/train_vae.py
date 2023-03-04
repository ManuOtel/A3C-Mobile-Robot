#### ---- IMPORTS AREA ---- ####
import torch, os, gc, cv2, time, random, warnings,  matplotlib

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from cbam_vae import VAE
#### ---- IMPORTS AREA ---- ####


#### ---- INIT AREA ---- ####
matplotlib.use( 'tkagg')
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
gc.collect()
unloader = transforms.ToPILImage()
#### ---- INIT AREA ---- ####


class Dataset(Dataset):
    def __init__(self, img_dir = './dataset5/'):

        self.img_path = []
        for i in os.listdir(img_dir):
            self.img_path.append(img_dir+i)
        
        self.img_path = []
        for i in os.listdir(img_dir):
            self.img_path.append(img_dir+i)

        self.length = len(self.img_path)
        self.transform = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_img = self.img_path[idx]

        img = cv2.imread(input_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_t = self.transform(img)

        return (img_t, img_t)


def set_all_seeds(seed=None) -> None:
    """Set all random seeds to a fixed value.

    :param seed: The seed to set. If None, then the seed is randomly initialized.

    :return: None"""
    if seed != None:
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('All random seed had been set to -> {}.'.format(seed))
    else:
        print('All seed have been toally randomly initialized.')


def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label='') -> None:
    """Plot the training loss.

    :param minibatch_losses: The minibatch losses.
    :param num_epochs: The number of epochs.
    :param averaging_iterations: The number of iterations to average over.
    :param custom_label: A custom label to add to the plot.

    :return: None"""
    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) <= 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:]) * 1.5  # len(minibatch_losses) = 1000, num_losses 1000
    ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations, ) / averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device) -> float:
    """Compute the loss for a given model and data loader.

    :param model: The model to compute the loss for.
    :param data_loader: The data loader to compute the loss for.
    :param loss_fn: The loss function to use.
    :param device: The device to use.

    :return: The loss for the model and data loader."""
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for pre_obs, _ in data_loader:
            pre_obs = pre_obs.to(device)
            logits = model(pre_obs)
            loss = loss_fn(logits, pre_obs, reduction='sum')
            num_examples += pre_obs.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def train(num_epochs, 
          model, 
          optimizer, 
          device, 
          train_loader, 
          loss_fn=None, 
          logging_interval=100,
          skip_epoch_stats=False, 
          reconstruction_term_weight=1, 
          save_model=None) -> dict:
    """Train the VAE model.

    :param num_epochs: The number of epochs to train for.
    :param model: The model to train.
    :param optimizer: The optimizer to use.
    :param device: The device to use.
    :param train_loader: The training data loader.
    :param loss_fn: The loss function to use.
    :param logging_interval: The interval to log the training loss.
    :param skip_epoch_stats: Whether to skip the epoch stats.
    :param reconstruction_term_weight: The weight of the reconstruction term.
    :param save_model: The path to save the model to.

    :return: The training log."""
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}
    if loss_fn is None:
        loss_fn = F.mse_loss
        #loss_fn = F.cross_entropy

    start_time = time.time()
    record_batch_cnt = 0
    for epoch in range(num_epochs):
        print('epoch: ', epoch + 1)
        model.train()
        # for batch_idx, (X_img, X_a, Y) in enumerate(train_loader):
        for batch_idx, (X_img, Y) in enumerate(train_loader):
            X_img = X_img.to(device)
            # X_a = X_a.to(device)
            Y = Y.to(device)

            # FORWARD AND BACK PROP
            z_a, z_mean, z_log_var, X_decoded = model(X_img)

            # total loss = reconstruction loss + KL divergence
            # kl_divergence = (0.5 * (z_mean**2 +
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var),
                                      axis=1)  # sum over latent dimension
            kl_div_buffer = kl_div
            batchsize = kl_div.size(0)
            kl_div = kl_div.mean()  # average over batch dimension

            pixelwise = loss_fn(X_decoded, Y, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
            pixelwise = pixelwise.mean()  # average over batch dimension

            loss = reconstruction_term_weight * pixelwise + kl_div
            # writer.add_scalar('Loss/Epoch', loss, epoch)

            optimizer.zero_grad()
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())

            if not batch_idx % logging_interval:  # 每n次print出紀錄
                record_batch_cnt += 1
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f   (pixelwise: %.4f | ki_div: %.4f)'
                      % (epoch + 1, num_epochs, batch_idx,
                        len(train_loader), loss, pixelwise, kl_div))
                # print('z_mean:', z_mean)
                # print('z_log_var: ', z_log_var)
                # print('kl_div_buffer: ', kl_div_buffer)
                # print('Exp z_log_var: ', torch.exp(z_log_var))

                # writer.add_scalar('Loss/batch_cnt', loss, record_batch_cnt)
                # writer.add_scalar('Reconstruct Loss/batch_cnt', pixelwise, record_batch_cnt)
                # writer.add_scalar('KL Div Loss/batch_cnt', kl_div, record_batch_cnt)

        writer.add_scalar('Loss/epoch', loss, epoch + 1)
        writer.add_scalar('Reconstruct Loss/epoch', pixelwise, epoch + 1)
        writer.add_scalar('KL Div Loss/epoch', kl_div, epoch + 1)

        print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f   (pixelwise: %.4f | ki_div: %.4f)'
              % (epoch + 1, num_epochs, batch_idx + 1,
                 len(train_loader), loss, pixelwise, kl_div))

        if epoch%100==0:
            torch.save(model.state_dict(), 'temp_vae_models/temp_model_e{}_l{}.pt'.format(epoch, loss))


        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference

                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                    epoch + 1, num_epochs, train_loss))
                log_dict['train_combined_per_epoch'].append(train_loss.item())

        #print('z_log_var: ', z_log_var)
        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    return log_dict


def plot_generated_images(data_loader, 
                          model, 
                          device, 
                          unnormalizer=None, 
                          figsize=(20, 2.5), 
                          n_images=10, 
                          modeltype='autoencoder') -> None:
    """Plots n_images generated images using the model and the data_loader.
    :param data_loader: DataLoader for the images
    :param model: Model to generate the images
    :param device: Device to use
    :param unnormalizer: Unnormalizer to unnormalize the images
    :param figsize: Figure size
    :param n_images: Number of images to plot
    :param modeltype: Type of model (autoencoder or VAE)
    
    :return: None"""

    _, axes = plt.subplots(nrows=2, ncols=n_images, sharex=True, sharey=True, figsize=figsize)

    for batch_idx, (cur_obs, _) in enumerate(data_loader):  # _: pre_obs (Y)

        cur_obs = cur_obs.to(device)
        # pre_act = pre_act.to(device)

        color_channels = cur_obs.shape[1]
        image_height = cur_obs.shape[2]
        image_width = cur_obs.shape[3]

        with torch.no_grad():
            if modeltype == 'autoencoder':
                # decoded_images = model(cur_obs, pre_act)[:n_images]
                decoded_images = model(cur_obs)[:n_images]
            elif modeltype == 'VAE':
                # z_a, z_mean, z_log_var, decoded_images = model(cur_obs, pre_act)[:n_images]
                z_a, z_mean, z_log_var, decoded_images = model(cur_obs)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = cur_obs[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                # curr_img = curr_img.transpose(0, 2)
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')


def plot_images_sampled_from_vae(model, device, latent_za_size=64, unnormalizer=None, num_images=10) -> None:
    """Plots images sampled from the VAE.

    :param model: VAE model
    :param device: Device to use
    :param latent_za_size: Size of the latent space
    :param unnormalizer: Unnormalizer to unnormalize the images
    :param num_images: Number of images to plot

    :return: None"""
    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################

        rand_za_code = torch.randn(num_images, latent_za_size).to(device)
        # new_fmap = model.za_fmap(rand_za_code)  # code to feature map
        new_fmap = model.z_fmap(rand_za_code)  # code to feature map
        new_images = model.decoder(new_fmap)  # featrue map to img
        color_channels = new_images.shape[1]
        image_height = new_images.shape[2]
        image_width = new_images.shape[3]

        ##########################
        ### VISUALIZATION
        ##########################

        image_width = 28

        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 2.5), sharey=True)
        decoded_images = new_images[:num_images]

        for ax, img in zip(axes, decoded_images):
            curr_img = img.detach().to(torch.device('cpu'))
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))  # [3, H, W] to RGB [H, W, 3]
                ax.imshow(curr_img)
            else:
                ax.imshow(curr_img.view((image_height, image_width)), cmap='binary')


if __name__ == '__main__':
    # Device
    CUDA_DEVICE_NUM = 0
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)

    # Hyperparameters
    RANDOM_SEED = None
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10000
    set_all_seeds(RANDOM_SEED)
    BATCH_SIZE = 128

    VAE_z_dim = 64
    Save_model_name = 'vae_models/VAE_'+str(VAE_z_dim)+'z'+'_batch'+str(BATCH_SIZE)+'.pt'

    # Load Dataset

    PATH_to_log_dir = './my_runs/VAE-z{}b{}lr{}'.format(VAE_z_dim, BATCH_SIZE, LEARNING_RATE)
    writer = SummaryWriter(PATH_to_log_dir)

    train_loader = torch.utils.data.DataLoader(dataset=Dataset(),  # torch TensorDataset format
                                               batch_size=BATCH_SIZE,  # mini batch size
                                               shuffle=True,
                                               num_workers=14)
    
    model = VAE(VAE_z_dim)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    log_dict = train(num_epochs=NUM_EPOCHS, model=model,
                            optimizer=optimizer, device=DEVICE,
                            train_loader=train_loader,
                            skip_epoch_stats=True,
                            logging_interval=10,  # logging_interval: 每n次print出目前執行紀錄
                            save_model=Save_model_name)

    # plot training loss
    plot_training_loss(log_dict['train_reconstruction_loss_per_batch'], NUM_EPOCHS, custom_label=" (reconstruction)")
    plot_training_loss(log_dict['train_kl_loss_per_batch'], NUM_EPOCHS, custom_label=" (KL)")
    plot_training_loss(log_dict['train_combined_loss_per_batch'], NUM_EPOCHS, custom_label=" (combined)")
    plt.show(block=False)

    # unnormalizer = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    plot_generated_images(data_loader=train_loader,
                          model=model,
                          # unnormalizer=unnormalizer,
                          device=DEVICE,
                          modeltype='VAE')

    # for i in range(10):
    for i in range(1):
        plot_images_sampled_from_vae(model=model, device=DEVICE, latent_za_size=VAE_z_dim)
        plt.show(block=False)

    plt.show()
    print('Debug')