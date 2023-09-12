import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import os 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchsummary import summary

import process
import loader
import network


def train(model, trainloader, valloader,device,optimizer, num_epoch=10):  # Train the model
    print("Start training...")
    trn_loss_hist = []
    trn_loss_hist_fine = []
    trn_acc_hist = []
    val_acc_hist = []
    model.train()  # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        print(f'-----------------Epoch = {i+1} / {num_epoch} -----------------' )
        for batch, label in tqdm(trainloader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            # This will call Network.forward() that you implement
            pred = model(batch)
            loss = nn.functional.mse_loss(pred,label) # criterion(pred, label)  # Calculate the loss
            running_loss.append(loss.item())
            loss.backward()  # Backprop gradients to all tensors in the network
            optimizer.step()  # Update trainable weights
            trn_loss_hist_fine.append(loss.item())
        print("\n Epoch {} loss:{}".format(i+1, np.mean(running_loss)))

        # Keep track of training loss, accuracy, and validation loss
        trn_loss_hist.append(np.mean(running_loss))
        trn_acc_hist.append(evaluate(model, trainloader,device=device))
        print("\n Evaluate on validation set...")
        val_acc_hist.append(evaluate(model, valloader,device=device))
    print("Done!")
    return trn_loss_hist,trn_loss_hist_fine, trn_acc_hist, val_acc_hist


def evaluate(model, loader,device):  # Evaluate accuracy on validation / test set
    model = model.to(device) 
    model.eval()  # Set the model to evaluation mode
    correct = 0
    running_acc = []
    with torch.no_grad():  # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            loss = nn.functional.mse_loss(pred,label) # criterion(pred, label)  # Calculate the loss
            running_acc.append(loss.item())
        acc = np.mean(running_acc)
        print("\n Evaluation accuracy: {}".format(acc))
        return acc

def main():
    if torch.cuda.is_available():
        print("Using the GPU. You are good to go!")
        device = 'cuda'
    elif torch.backends.mps.is_available():
        print("Using the mps. You are good to go!")
        device = 'mps'
    else:
        print("Using the CPU. Overall speed may be slowed down")
        device = 'cpu'

    # Load the dataset and train, val, test splits
    print("Loading datasets...")
    # Transform from [0,255] uint8 to [0,1] float,
    # then normalize to zero mean and unit variance

    # data loading
    currdir = os.getcwd() 
    train_dir = os.path.join(currdir,'train/images')
    val_dir = os.path.join(currdir,'val/images')

    
    train_imgs = loader.image_loader(train_dir, 10000)
    val_test_imgs = loader.image_loader(val_dir, 6000)

    train_imgs_resized = process.resize_to_shape(train_imgs,(64,64))
    val_test_imgs_resized = process.resize_to_shape(val_test_imgs,(64,64))

    val_imgs = val_test_imgs_resized[:3000]
    test_imgs = val_test_imgs_resized[3000:]

    print("Done!")

    train_L = process.imgs_tensor(train_imgs_resized)
    val_L = process.imgs_tensor(val_imgs)
    test_L = process.imgs_tensor(test_imgs)

    # Create dataloaders
    # Experiment with different batch sizes
    batch_size=350
    trainloader = DataLoader(train_L, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_L, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_L, batch_size=batch_size, shuffle=True)

    # Load model
    dropout=0.3
    model = network.Small(
            dropout=dropout, 
            kernel_size=3,
            downscale_by=2,
            batch_norm=True,
            activation=torch.nn.ReLU
        )
    network_channels = [32,64,128]

    # model = network.UNet(
    #     in_channels=1,
    #     layer_channels=network_channels,
    #     out_channels=[2,64,128],
    #     downsample_scale=3,
    #     kernel_size=3,
    #     n_convs_per_layer = 2, 
    #     dropout=0.3
    # )
    print('Your network:')
    print(summary(model, (1,28,28))) # visualize your model



    # Set up optimization hyperparameters
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epoch = 15
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                        weight_decay=weight_decay)

    model.train()
    trn_loss_hist,fine_trn_losses, trn_acc_hist, val_acc_hist = (
        train(model=model.to(device), 
             trainloader=trainloader,
             valloader=valloader,
             optimizer=optimizer,
             device=device,
             num_epoch=num_epoch)
    )
    # Note down the evaluation accuracy on test set
    print("\n Evaluate on test set")
    evaluate(model.to(device), testloader,device=device)
    model = model.cpu()
    if type(model) is network.UNet: 
        model_name = f'UNet_{learning_rate:1.2e}_epoch_{num_epoch}_dropout_{dropout}_{"_".join(str(ch) for ch in network_channels)}'
    else: 
        model_name = f'small_{learning_rate:1.2e}_epoch_{num_epoch}_dropout_{dropout}'

    torch.save(model,f'model_saves/{model_name}.pt')
    # Submit the accuracy plot
    # visualize the training / validation accuracies
    plt.figure()
    plt.plot(np.array(range(len(fine_trn_losses)))*(len(train_imgs)/num_epoch),fine_trn_losses)
    x = np.arange(num_epoch)
    plt.plot(x, trn_acc_hist)
    plt.plot(x, val_acc_hist)
    plt.legend(['training_fine','training', 'Validation'])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Image Colorization Accuracy by SSIM')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('figures/training_{model_name}.png', dpi=300)
    plt.show()

    model.eval()
    sample = loader.load_img('val/images/val_9991.JPEG')
    tens_l_orig, tens_l_rs,img_ab = process.preprocess_img(sample, HW=(64,64), resample=3)
    tens_l_rs = tens_l_rs#.to(device)

    img_bw = process.postprocess(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img = process.postprocess(tens_l_orig, model(tens_l_rs).cpu())

    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    plt.imshow(sample)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(out_img)
    plt.title('Output Colorized')
    plt.axis('off')
    plt.savefig('figures/image_{model_name}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    main()