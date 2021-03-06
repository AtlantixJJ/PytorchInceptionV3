"""
A script to test Pytorch and Tensorflow InceptionV3 have consistent behavior.
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import sys, argparse, os, pathlib
sys.path.insert(0, ".")
import numpy as np
import tensorflow as tf
import torch
from inception_origin import inception_v3
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="", help="The path to data dir.")
parser.add_argument("--load_path", default="", help="The path to changed pytorch inceptionv3 weight. Run change_statedict.py to obtain.")
args = parser.parse_args()

def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)

def torch2numpy(x):
    return x.detach().cpu().numpy().transpose(0, 2, 3, 1)

def diffmap(x, y):
    energy = (np.abs(x).mean() + np.abs(y).mean()) / 2
    absdiff = np.abs(x - y).mean(3).mean(0)
    norm = absdiff / energy
    return norm

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

data_dir = args.data_path
if len(data_dir) > 0:
    print("=> Read images from %s" % data_dir)
    imgs_pil = [Image.open(open(data_dir + s, "rb")).resize((299,299), Image.BILINEAR) for s in os.listdir(data_dir)]
    imgs = [np.asarray(img).astype("float32") for img in imgs_pil]
else:
    print("=> Generate from random")
    
x_arr = np.array(imgs)
x_arr_tf = x_arr
# TF InceptionV3 graph use [0, 255] scale image
feed = {'FID_Inception_Net/ExpandDims:0': x_arr_tf}
# This is identical to TF image transformation
x_arr_torch = (x_arr - 128) * 0.0078125
x_torch = torch.from_numpy(x_arr_torch.transpose(0, 3, 1, 2)).float().cuda()

model = inception_v3(pretrained=True, aux_logits=False, transform_input=False)
if len(args.load_path) > 1:
    # default: pretrained/inception_v3_google.pth
    print("=> Get changed weight from %s" % args.load_path)
    try:
        model.load_state_dict(torch.load(args.load_path))
    except RuntimeError as e:
        print(e)
        print("!> Load failed")
        pass
model.cuda()
model.eval()

if x_torch.size(2) != 299:
    import torch.nn.functional as F
    x_torch = F.interpolate(x_torch,
            size=(299, 299),
            mode='bilinear',
            align_corners=False)
features = model.get_feature(x_torch)
feature_pytorch = features[-1].detach().cpu().numpy()
if len(feature_pytorch.shape) == 4:
    feature_pytorch = feature_pytorch[:, :, 0, 0]

inception_path = check_or_download_inception("pretrained")
with tf.gfile.FastGFile("pretrained/classify_image_graph_def.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

layername = "FID_Inception_Net/pool_3:0"
layer = sess.graph.get_tensor_by_name(layername)
ops = layer.graph.get_operations()
for op_idx, op in enumerate(ops):
    for o in op.outputs:
        shape = o.get_shape()
        if shape._dims != []:
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            # print(o.name, shape, new_shape)
            o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

tensor_list = [n.name for n in tf.get_default_graph().as_graph_def().node]

target_layer_names = [
    "FID_Inception_Net/Mul:0",
    "FID_Inception_Net/conv:0",
    "FID_Inception_Net/conv_1:0",
    "FID_Inception_Net/conv_2:0",
    "FID_Inception_Net/conv_3:0",
    "FID_Inception_Net/conv_4:0",
    "FID_Inception_Net/mixed/join:0",
    "FID_Inception_Net/mixed_1/join:0",
    "FID_Inception_Net/mixed_2/join:0",
    "FID_Inception_Net/mixed_3/join:0",
    "FID_Inception_Net/mixed_4/join:0",
    "FID_Inception_Net/mixed_5/join:0",
    "FID_Inception_Net/mixed_6/join:0",
    "FID_Inception_Net/mixed_7/join:0",
    "FID_Inception_Net/mixed_8/join:0",
    "FID_Inception_Net/mixed_9/join:0",
    "FID_Inception_Net/mixed_10/join:0",
    "FID_Inception_Net/pool_3:0"]
target_layers = [sess.graph.get_tensor_by_name(l) for l in target_layer_names]

sess.run(tf.global_variables_initializer())
res = sess.run(target_layers, feed)
x_tf = res[0]
feature_tensorflow = res[-1][:, 0, 0, :]

print("=> Pytorch pool3:")
print(feature_pytorch[0][:6])
print("=> Tensorflow pool3:")
print(feature_tensorflow[0][:6])
print("=> Mean abs difference")
print(np.abs(feature_pytorch - feature_tensorflow).mean())

pytorch_layers = [torch2numpy(f) for f in features[1:-1]]
tensorflow_layers = res[1:-1]
rads = []
print(len(pytorch_layers), len(tensorflow_layers))
for i in range(len(pytorch_layers)):
    print("=> Layer %d %s" % (i+1, target_layer_names[i+1]))
    print("%s\t%s" % (str(pytorch_layers[i].shape), str(tensorflow_layers[i].shape)))
    diff = diffmap(pytorch_layers[i], tensorflow_layers[i])
    rads.append(diff.mean())
    print(rads[-1])
    if rads[-1] > 0.01:
        diff_image = (diff - diff.min()) / (diff.max() - diff.min())
    else:
        diff_image = diff
    Image.fromarray((diff_image * 255).astype("uint8")).resize((64, 64)).save(open("layer_%d.png" % (i+1), "wb"), format="PNG")

fig = plt.figure(figsize=(8,8.5))
plt.bar([name[18:][:-2].replace("/join", "") for name in target_layer_names[1:-1]], rads)
plt.xlabel("Layer Name")
plt.xticks(rotation=60)
plt.ylabel("Relative Absolute Difference")
plt.savefig("rabsdiff.png")
plt.close()

def get_tf_layer(name):
    return sess.run(sess.graph.get_tensor_by_name(name + ':0'), feed)



