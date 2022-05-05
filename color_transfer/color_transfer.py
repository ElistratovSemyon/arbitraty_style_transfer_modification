import numpy as np
import torch


def histogram_matching(input_img, ref_img, alpha=1.0):
    """
    Histogram color transfer function.

    Transform input image colors to reference colors.

    Parameters
    ----------
    input_img : torch.tensor
        Input image. Has 3 dimensions: HxWxC.

    ref_img : torch.tensor
        Reference image. Has 3 dimensions: HxWxC.

    alpha : float
        Strength of color transfer. 
        alpha in [0.0, 1.0].
    Returns
    -------
    res : torch.tensor
        Result image.

    """
    res = np.zeros_like(input_img)
    for channel in range(3):

        n_src = len(input_img[:, :, channel].ravel())
        n_ref = len(ref_img[:, :, channel].ravel())

        src_values, src_unique_indices, src_counts = torch.unique(torch.tensor(
            input_img[:, :, channel].ravel()), return_inverse=True,
            return_counts=True)
        src_quantiles = torch.cumsum(
            src_counts, 0) / input_img[:, :, channel].ravel().shape[0]
        src_quantiles = src_quantiles.cpu().numpy()
        ref_values, ref_counts = torch.unique(torch.tensor(
            ref_img[:, :, channel].ravel()), return_counts=True)

        mask_src = np.zeros(256)
        mask_src[src_values] = src_counts / n_src

        mask_ref = np.zeros(256)
        mask_ref[ref_values] = ref_counts / n_ref

        mean_mask = alpha * mask_ref + (1-alpha) * mask_src
        mean_mask = mean_mask.cumsum()

        ref_values = ref_values.cpu().numpy()
        src_unique_indices = src_unique_indices.cpu().numpy()
        interp_a_values = np.rint(
            np.interp(src_quantiles, mean_mask, np.arange(256))).astype(np.uint8)
        res[:, :, channel] = interp_a_values[src_unique_indices].reshape(
            input_img[:, :, channel].shape)
    return res
