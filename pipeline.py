import torch
import torch.nn as nn
from color_transfer.color_transfer import histogram_matching


class Pipeline:
    """
    Arbitrary style transfer pipeline.

    Allow controllable stylization with color transfer.

    Parameters
    ----------
    model : nn.Module
        Style transfer module.

    device : str of torch.device
        Type of computational core.
    
    theta : float or None
        Threeshold for stylization strength. 
        If alpha less than theta, model perform interpolation 
        between content and stylization result.
        If theta is None - no interpolation.
    """
    def __init__(self, model, device, theta=None):
        self.model = model.to(device)
        self.device = device
        self.theta = theta

    def __call__(self, content, style, alpha=1.0, beta=None, theta=None):
        """
        Perform stylization.

        Parameters
        ----------
        content : torch.tensor
            Batch of content images.

        style : torch.tensor
            Batch of style images.
        
        alpha : float
            Stylization strength. Greater alpha - more stylization.
            alpha in [0, 1].

        beat : float
            Color transfer strength. Greater beta - more color transfer.
            beta in [0, 1].
        
        theta : float or None
            Threeshold for stylization strength. 
            If alpha less than theta, model perform interpolation 
            between content and stylization result.
            If theta is None - no interpolation.
        """
        if (content.ndim != 4) or (style.ndim != 4):
            raise TypeError("Images must have 4 dimensions.")
        res = self.model(content.to(self.device), style.to(self.device), alpha)
        if not theta is None:
            self.theta = theta
        if not self.theta is None:
            if alpha < self.theta:
                res = (res * alpha + (self.theta - alpha)
                       * content) / self.theta
        if not beta is None:
            for i in range(res.shape[0]):
                content_color = (content[i].permute(
                    (1, 2, 0)).cpu().numpy() * 255).astype(int)
                res_color = (res[i].permute((1, 2, 0)).cpu().numpy()
                     * 255).astype(int)
                tmp = histogram_matching(res_color, content_color, beta)
                res[i] = torch.tensor(tmp).permute((2, 0, 1)).to(self.device) / 255

        return res
