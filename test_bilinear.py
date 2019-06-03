"""
Pytorch and Tensorflow bilinear interpolation is different.
"""
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

img = Image.open(open("data/cifar10_test/00554.jpg", "rn")).convert("RGB")
img_arr = np.asarray(img) # (32, 32, 3), 255

x_tf = (img_arr.astype("float32") - 127.5) / 127.5 # (32, 32, 3), 255
x_torch = torch.from_numpy(x_tf.transpose(2, 0, 1))

x_tf = np.expand_dims(x_tf, 0)
x_torch = x_torch.unsqueeze(0)

print("=> Input size (Tensorflow, Pytorch):")
print("=> %s\t%s" % (str(x_tf.shape), str(x_torch.shape)))

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    res_tf = tf.image.resize_bilinear(x, (299, 299)).eval({x:x_tf})[0]

res_torch = F.interpolate(x_torch, (299, 299), mode='bilinear')[0].detach().numpy().transpose(1, 2, 0)

res_tf = ((res_tf + 1) * 127.5).astype("uint8")
res_torch = ((res_torch + 1) * 127.5).astype("uint8")

print("=> Output size (Tensorflow, Pytorch):")
print("=> %s\t%s" % (str(res_tf.shape), str(res_torch.shape)))

img_pil = img.resize((299, 299), Image.BILINEAR)
res_pil = np.asarray(img_pil)
img_pil.save(open("figures/pil.png", "wb"), format="PNG")

diff_tf = np.abs((res_pil.astype("float32") - res_tf.astype("float32")).mean(2))
diff_torch = np.abs((res_pil.astype("float32") - res_torch.astype("float32")).mean(2))
r, l = max(diff_tf.max(), diff_torch.max()), min(diff_tf.min(), diff_torch.min())

def norm255(x):
    return (x * 255).astype("uint8")

diff_tf = norm255((diff_tf - l) / (r - l))
diff_torch = norm255((diff_torch - l) / (r - l))

print(diff_tf.max(), diff_tf.min(), diff_tf.shape)
Image.fromarray(diff_tf).save(open("figures/diff_tf.png", "wb"), format="PNG")
Image.fromarray(diff_torch).save(open("figures/diff_pytorch.png", "wb"), format="PNG")

Image.fromarray(res_tf).save(open("figures/tf.png", "wb"), format="PNG")
Image.fromarray(res_torch).save(open("figures/pytorch.png", "wb"), format="PNG")