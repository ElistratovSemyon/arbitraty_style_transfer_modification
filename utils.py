import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from style_transfer_module.vggScoring import VggScoring
from style_transfer_module.stylizationModel import StylizationModel

from PIL import Image
import os
from os.path import join, dirname

def postprocess_image(data):
    """ Transform tensor to PIL.Image. """
    img = (255*data).cpu().squeeze(0).numpy()  # data in [0,1] (output of sigmoid)
    img = img.transpose(1, 2, 0).astype("uint8")  # CxHxW -> HxWxC for PIL image
    img = Image.fromarray(img)
    return img

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1,-1, 1, 1)  # new tensor with the same dtyp`e and device as batch
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1,-1, 1, 1)   # .view(1,-1, 1, 1) makes [1,3,1,1] shape of tensor
    #batch = batch.div_(255.0) #to_tensor automatically scale to [0, 1]
    return (batch - mean) / std

def gram_matrix(y):
    """ Compute gram matrix for 4 dimensional image. """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)  # [batches(=1),channels,w*h]->[batches(=1),w*h,channels]
    gram = features.bmm(features_t) / (ch * h * w)   # result is [batches(=1), channles, channels]
    return gram

def train_model(content_loader, main_styles_loader, base_styles_dataset=None, 
                dropout=None, connections=True,
                checkpoint=None, save_dir="./checkpoints/", 
                lr=1e-4, sample_alpha=True, 
                max_iter=80000, prob_base_style=0.1,
                device=None):
    """
    Train style transfer model.

    Parameters
    ----------
    content_loader : DataLoader
        DataLoader for content dataset.

    main_styles_loader : DataLoader
        DataLoader for style dataset.
        
    base_styles_dataset : torchvision.dataset
        Dataset of base styles.
    
    dropout : float
        Value of probability for style transfer model.
        
    connections : bool
        Use additional connections from encoder to decoder in style transfer model.
        
    checkpoint : str
        Path to checkpoint.
        
    save_dir : str
        Path for the directory where to save the model.
    
    lr : float
        Value of learning rate for optimizer.
        
    sample_alpha : bool
        Sample alpha from uniform distribution during training.
     
    max_iter : int
        Number of iteration.
        
    prob_base_style : float
        Probability for mixing base styles.

    device : str of torch.device
        Type of computational core.
    """

    if device is None:
    	device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StylizationModel(dropout, connections).to(device)
    vgg = VggScoring()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    alpha = 1.0
    if not checkpoint is None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    model.to(device)
    vgg.to(device)
    
    content_losses = []
    style_losses = []
    total_losses = []
    
    n_epoch = 0
    img_count = 0
    
    
    while True:
        if (img_count >= max_iter):
            break
        style_loss_agg = 0
        content_loss_agg = 0
        batch_count = 0
        for i, ((x, _), (style, _)) in enumerate(zip(content_loader, main_styles_loader)):
            if (img_count >= max_iter):
                break
                
            if not base_styles_dataset is None:
                if np.random.binomial(1, prob_base_style) == 1:
                    ind = np.random.choice(len(base_styles_dataset))
                    style[0] = base_styles_dataset[ind][0]
                
            if sample_alpha:
                alpha = np.random.uniform()
            
            n_batch = len(x)
            x = x.to(device)
            style = style.to(device)
            img_count += n_batch
            batch_count += 1

            y = model(x, style, alpha)

            features_y = vgg(normalize_batch(y))
            features_x = vgg(normalize_batch(x))
            features_style = vgg(normalize_batch(style))
            gram_style = [gram_matrix(y) for y in features_style]

            content_loss = mse_loss(features_y.relu3_3, features_x.relu3_3)

            style_loss = 0.0
            n_w = 0
            # features_y are just tensor representations on style layers
            for ft_y, gm_s in zip(features_y, gram_style):
                # convert tensor representations of stylization on style layers to Gramm matrices
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y,
                                       gm_s[:n_batch, :, :])*style_weigth[n_w]
                n_w += 1     # sum gramm matrices tohether
            
            style_loss *= alpha
            total_loss = content_loss + style_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            style_loss_agg += style_loss
            content_loss_agg += content_loss
            if img_count % 250 == 0 and img_count > 1:
                print("--------------------------------")
                print("n_epoch %d, img_count %d, loss %.4f" %
                      (n_epoch, img_count, total_loss))
                fig = plt.figure(figsize=(10, 10))
                plt.title("Result")
                plt.imshow(y.cpu().detach()[0].permute((1, 2, 0)))
                plt.show()
                fig = plt.figure(figsize=(10, 10))
                plt.title("Style")
                plt.imshow(style.cpu().detach()[0].permute((1, 2, 0)))
                plt.show()
                print("--------------------------------")
                if img_count % 1000 == 0:
                    torch.save({
                        'epoch': n_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': content_loss,
                    }, os.path.join(save_dir, "model_%d_%d.pth" % (n_epoch, img_count)))
        n_epoch += 1
        print("Epoch: %d" % n_epoch)
        print("Style loss: %.4f" % (style_loss_agg / batch_count))
        print("Content loss: %.4f" % (content_loss_agg / batch_count))


def build_model(dropout=None, connections=True, checkpoint=None):
    """
    Train style transfer model.

    Parameters
    ----------
    dropout : float
        Value of probability for style transfer model.
        
    connections : bool
        Use additional connections from encoder to decoder in style transfer model.
        
    checkpoint : str
        Path to checkpoint.
    """
    model = StylizationModel(dropout, connections)
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu'))["model_state_dict"])
    return model


def eval_pipeline(model, content_list, style_dataset, output_dir, max_images=40,
               alpha_list=torch.linspace(0.0, 1.0, 11), 
               beta_list=torch.linspace(0.0, 1.0, 11)):
    """
    Evaluate style transfer pipeline.

    Parameters
    ----------
    model : Pipeline class
        Style transfer pipeline.
        
    content_list : list
        List of content images.

    style_dataset : DataLoader
        DataLoader for style dataset.
        
    output_dir : str
        Path for the directory where to save the stylization results.
    
    max_images : int
        Maximum number of images to stylize.
        
    alpha_list : array-like
        Array of parameters for control stylization strength.
     
    beta_list : array-like
        Array of parameters for control color trasfer strength.
    """
    i = 0
    for style, _ in style_dataset:
        with torch.no_grad():
            for content in content_list:

                if i > max_images:
                    break
                i += 1

                path = os.path.join(output_dir, "%d_image" % i)
                os.makedirs(path, exist_ok=True)
                postprocess_image(content).save(
                    os.path.join(path, "content.png"), "PNG")
                postprocess_image(style).save(
                    os.path.join(path, "style.png"), "PNG")

                for alpha in alpha_list:
                    for beta in beta_list:
                        img = model(content, style, alpha, beta)
                        
                        res_path = "%d_alpha_%d_beta.png" % (alpha * 100, beta * 100)
                        postprocess_image(img).save(os.path.join(path, res_path), "PNG")