import torch
import numpy as np
import torch.nn as nn
import random
import cv2
from torch.utils.data import Dataset
import h5py

def matrix_rela(x, t):
    soft = nn.Softmax(dim=-1)
    attn = x @ x.transpose(-2, -1)
    attn_soft = soft(attn)
    attn_exp = torch.exp(attn_soft/t)
    return attn_exp

class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            '''这里修改了return的返回值'''
            return np.array(f['lr'][str(idx)]), np.array(f['hr'][str(idx)])

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

# BSRN中带的方法
def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            0. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 0].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]
    # 多一步转换 转换到dtype范围 所以从16. -》 0-0
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        # height, width = X.shape[2], X.shape[3]
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 255.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (65.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 255.
    else:
        raise Exception('Unknown Type', type(img))

def convert_rgb_to_y_multi(img):
    if type(img) == np.ndarray:
        return 16. + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 255.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            # out = torch.zeros((img.shape[0], 0, img.shape[2], img.shape[3]))
            for i in range(img.shape[0]):
                img[i, 0, :, :] = 16. + (65.738 * img[i, 0, :, :] + 129.057 * img[i, 1, :, :] + 25.064 * img[i, 2, :, :]) / 255.
        return img[:, 0, :, :].unsqueeze(1)
    else:
        raise Exception('Unknown Type', type(img))

# img = torch.rand((16, 3, 64, 64))
# print(convert_rgb_to_y_multi(img).shape)
def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 255.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 255.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 255.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (65.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 255.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 255.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 255.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 255. + 408.583 * img[:, :, 2] / 255. - 222.921
        g = 298.082 * img[:, :, 0] / 255. - 100.291 * img[:, :, 1] / 255. - 208.120 * img[:, :, 2] / 255. + 135.576
        b = 298.082 * img[:, :, 0] / 255. + 516.412 * img[:, :, 1] / 255. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 255. + 408.583 * img[2, :, :] / 255. - 222.921
        g = 298.082 * img[0, :, :] / 255. - 100.291 * img[1, :, :] / 255. - 208.120 * img[2, :, :] / 255. + 135.576
        b = 298.082 * img[0, :, :] / 255. + 516.412 * img[1, :, :] / 255. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def fun_convert(inputs):
    img = torch.zeros(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3])
    for i in range(inputs.shape[0]):
        img[i, 0, :, :] = 16. + (
                    65.738 * inputs[i, 0, :, :] + 129.057 * inputs[i, 1, :, :] + 25.064 * inputs[i, 2, :, :]) / 255.

    return img


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 0);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            0. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 0].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 0].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            0. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 0].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 0].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 0].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 0].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

def clone_cat(x, times):
    if len(x.size()) == 2:
        H, W = x.size()
        out = torch.zeros((times, H, W))
        for i in range(times):
            out[i, :, :] = x[:, :]
    elif len(x.size()) == 3:
        in_C, H, W = x.size()
        out = torch.zeros((times, in_C, H, W))
        for i in range(times):
            out[i, :, :, :] = x[:, :, :]

    return out

# class Sobel_layer(nn.Module):
#
#     def __init__(self, in_channel=64, out_channel=64, RGB_mode=False):
#         super(Sobel_layer, self).__init__()
#         self.mode = RGB_mode
#         kernel_v = [[0, -1, 0],
#                     [0, 0, 0],
#                     [0, 1, 0]]
#         kernel_h = [[0, 0, 0],
#                     [-1, 0, 1],
#                     [0, 0, 0]]
#         kernel_h = torch.FloatTensor(kernel_h)
#         kernel_v = torch.FloatTensor(kernel_v)
#         if RGB_mode is False:
#             kernel_h = clone_cat(clone_cat(kernel_h, in_channel), out_channel)
#             kernel_v = clone_cat(clone_cat(kernel_v, in_channel), out_channel)
#             self.weight_h = kernel_h
#             self.weight_v = kernel_v
#         else:
#             kernel_h = kernel_h.unsqueeze(0).unsqueeze(0)
#             kernel_v = kernel_v.unsqueeze(0).unsqueeze(0)
#             self.weight_h = kernel_h
#             self.weight_v = kernel_v
#
#     def get_gray(self,x):
#         '''
#         Convert image to its gray one.
#         '''
#         gray_coeffs = [65.738, 129.057, 25.064]
#         convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#         x_gray = x.mul(convert).sum(dim=1)
#         return x_gray.unsqueeze(1)
#
#     def forward(self, x):
#         if self.mode is True:
#             x = self.get_gray(x)
#
#         x_v = F.conv2d(x, self.weight_v, padding='same')
#         x_h = F.conv2d(x, self.weight_h, padding='same')
#         x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
#
#         return x


def data_Augmentation(lr, hr, p=0.1):
    '''
    lr, hr : (batch_size, channel, height, width)
    '''
    gamma = random.random()
    beta = random.uniform(p, 1)
    # 每个batch都重新随机生成一个beta
    # 不变
    if p >= gamma:
        lr = lr
        hr = hr

    thorldvalue = (1-p) / 4
    if p < gamma:
        # 旋转90
        if 0 < beta <= thorldvalue:
            for i in range(lr.size(0)):
                lr[i, :, :, :] = torch.rot90(lr[i, :, :, :].squeeze(0), dims=(2, 1)).unsqueeze(0)
                hr[i, :, :, :] = torch.rot90(hr[i, :, :, :].squeeze(0), dims=(2, 1)).unsqueeze(0)
            # print(thorldvalue, beta,'旋转90')
        # 旋转180
        elif thorldvalue < beta <= 2 * thorldvalue:
            for i in range(lr.size(0)):
                lr[i, :, :, :] = torch.rot90(lr[i, :, :, :].squeeze(0), 2, dims=(2, 1)).unsqueeze(0)
                hr[i, :, :, :] = torch.rot90(hr[i, :, :, :].squeeze(0), 2, dims=(2, 1)).unsqueeze(0)
            # print(thorldvalue, beta,'旋转180')
        # 旋转270
        elif 2 * thorldvalue < beta <= 3 * thorldvalue:
            for i in range(lr.size(0)):
                lr[i, :, :, :] = torch.rot90(lr[i, :, :, :].squeeze(0), 3, dims=(2, 1)).unsqueeze(0)
                hr[i, :, :, :] = torch.rot90(hr[i, :, :, :].squeeze(0), 3, dims=(2, 1)).unsqueeze(0)
            # print(thorldvalue, beta,'旋转270')
        # 水平翻转
        elif 3 * thorldvalue < beta <= 4 * thorldvalue:
            for i in range(lr.size(0)):
                lr[i, :, :, :] = torch.flip(lr[i, :, :, :].squeeze(0), dims=[2]).unsqueeze(0)
                hr[i, :, :, :] = torch.flip(hr[i, :, :, :].squeeze(0), dims=[2]).unsqueeze(0)
            # print(thorldvalue, beta,'翻转')

    return lr, hr


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class FFTloss(nn.Module):
    def __init__(self, loss_weight=0.05, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)


    def forward(self, pred, target):

        l1 = self.criterion(pred, target)
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft) + l1