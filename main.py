import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import DataLoader

from dataset.dataset import ToothDataset, ToothDataset_3Ch
from model.p2p import Discriminator, GeneratorUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(): #input: mask, gap, opp, finger, gro
    loss_func_gan = nn.BCELoss()
    loss_func_pix = nn.L1Loss()

    lambda_pixel = 100

    patch = (1,16,16)

    model_dis = Discriminator().to(device)
    model_gen = GeneratorUNet().to(device)

    from torch import optim
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999

    opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))

    model_dis.train()
    model_gen.train()

    batch_count = 0
    best_loss = 100
    epochs = 3000
    loss_dict = {'gen':[], 'dis':[]}

    train_dataset = ToothDataset(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        batch_count = 0
        avg_gen_loss = 0
        avg_dis_loss = 0
        for image, label, _ in train_loader:
            image = image.to(device)
            input = image[:,-1,:,:]
            input_shape = input.shape
            input = input.reshape(input_shape[0], 1, input_shape[1], input_shape[2])
            label = label.to(device)

            real_label = torch.ones(label.size(0), *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(label.size(0), *patch, requires_grad=False).to(device)

            #train discriminator
            model_gen.zero_grad()

            fake_image = model_gen(image)

            out_dis = model_dis(fake_image, label)

            gen_loss = loss_func_gan(out_dis, real_label)
            pixel_loss = loss_func_pix(fake_image, label)

            g_loss = gen_loss + pixel_loss * lambda_pixel
            g_loss.backward()
            opt_gen.step()

            #train generator
            model_dis.zero_grad()

            out_dis = model_dis(label, input)
            real_loss = loss_func_gan(out_dis, real_label)

            out_dis = model_dis(fake_image.detach(), input)
            fake_loss = loss_func_gan(out_dis, fake_label)

            d_loss = (real_loss + fake_loss) /2.
            d_loss.backward()
            opt_dis.step()

            avg_gen_loss += g_loss.item()
            avg_dis_loss += d_loss.item()

            batch_count += 1

            if best_loss > g_loss.item():
                torch.save(model_gen.state_dict(), 'best_p2p_model_5ch.pt')
                print('Epoch: %.0f/%.0f, Batch: %.0f/32, G_Loss: %.6f, D_Loss: %.6f' %(epoch, epochs, batch_count+1, g_loss.item(), d_loss.item()))
                best_loss = g_loss.item()

        avg_gen_loss /= 32
        avg_dis_loss /= 32
        loss_dict['gen'].append(avg_gen_loss)
        loss_dict['dis'].append(avg_dis_loss)    
        print('Epoch: %.0f/%.0f, G_Loss: %.6f, D_Loss: %.6f' %(epoch, epochs, g_loss.item(), d_loss.item()))

    plt.figure(figsize=(10,5))
    plt.title('Loss Progress')
    plt.plot(loss_dict['gen'], label='Gen. Loss')
    plt.plot(loss_dict['dis'], label='Dis. Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('pix2pix_loss.png')

def main3Ch(): #input: mask, gap, opp
    loss_func_gan = nn.BCELoss()
    loss_func_pix = nn.L1Loss()

    lambda_pixel = 100

    patch = (1,16,16)

    model_dis = Discriminator(in_channels=3).to(device)
    model_gen = GeneratorUNet(in_channels=3, out_channels=3).to(device)

    from torch import optim
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999

    opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))

    model_dis.train()
    model_gen.train()

    batch_count = 0
    best_loss = 100
    epochs = 100
    loss_dict = {'gen':[], 'dis':[]}

    train_dataset = ToothDataset_3Ch(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        batch_count = 0
        avg_gen_loss = 0
        avg_dis_loss = 0
        for image, label, _ in train_loader:
            image = image.to(device)
            label = label.to(device)

            real_label = torch.ones(label.size(0), *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(label.size(0), *patch, requires_grad=False).to(device)

            #train discriminator
            model_gen.zero_grad()

            fake_image = model_gen(image)

            out_dis = model_dis(fake_image, label)

            gen_loss = loss_func_gan(out_dis, real_label)
            pixel_loss = loss_func_pix(fake_image, label)

            g_loss = gen_loss + pixel_loss * lambda_pixel
            g_loss.backward()
            opt_gen.step()

            #train generator
            model_dis.zero_grad()

            out_dis = model_dis(label, image)
            real_loss = loss_func_gan(out_dis, real_label)

            out_dis = model_dis(fake_image.detach(), image)
            fake_loss = loss_func_gan(out_dis, fake_label)

            d_loss = (real_loss + fake_loss) /2.
            d_loss.backward()
            opt_dis.step()

            avg_gen_loss += g_loss.item()
            avg_dis_loss += d_loss.item()

            batch_count += 1

            if best_loss > g_loss.item():
                torch.save(model_gen.state_dict(), 'best_p2p_model_3ch.pt')
                print('Epoch: %.0f/%.0f, Batch: %.0f/32, G_Loss: %.6f, D_Loss: %.6f' %(epoch, epochs, batch_count+1, g_loss.item(), d_loss.item()))
                best_loss = g_loss.item()
        avg_gen_loss /= 32
        avg_dis_loss /= 32
        loss_dict['gen'].append(avg_gen_loss)
        loss_dict['dis'].append(avg_dis_loss)
        print('Epoch: %.0f/%.0f, G_Loss: %.6f, D_Loss: %.6f' %(epoch, epochs, avg_gen_loss, avg_dis_loss))

    plt.figure(figsize=(10,5))
    plt.title('Loss Progress')
    plt.plot(loss_dict['gen'], label='Gen. Loss')
    plt.plot(loss_dict['dis'], label='Dis. Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('pix2pix_loss_3ch.png')
   

def load(ch=5):
    
    if ch == 5:
        model = GeneratorUNet()
        state = torch.load('best_p2p_model_5ch.pt')
        test_dataset = ToothDataset(is_train=False)
        save_path = os.path.join('./', 'pix2pix_results_5ch')
    else:
        model = GeneratorUNet(in_channels=ch, out_channels=ch)
        state = torch.load('best_p2p_model_{}ch.pt'.format(ch))
        test_dataset = ToothDataset_3Ch(is_train=False)
        save_path = os.path.join('./', 'pix2pix_results_3ch')
        
    model.load_state_dict(state)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        for input, label, image_name in test_loader:
            gen_image = model(input).detach().cpu()
            for i, image in enumerate(gen_image):
                image = image * 255
                image = image.numpy().astype(np.uint8)
                if ch == 5:
                    image = np.squeeze(image, axis=0)
                else:
                    image = image.transpose(1, 2, 0)
                image = Image.fromarray(image)
                image.save(os.path.join(save_path, image_name[i]))

if __name__ == '__main__':
    main()
    # main3Ch()
    load()