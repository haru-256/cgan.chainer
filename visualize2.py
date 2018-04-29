import chainer
from chainer import Variable
from chainer import serializers
from generator import Generator
from visualize import combine_images
import pathlib
import matplotlib.pyplot as plt

path = pathlib.Path("result")
abs_path = path.resolve()

gen = Generator()  # prepare model
serializers.load_npz(path/"gen_iter_46875.npz", gen) # load pretraining model
# serializers.load_npz(path/"gen_iter_1406250.npz", gen) # load pretraining model
xp = gen.xp  # get numpy or cupy

z = Variable(xp.zeros((10, 100)).astype("f"))  # make noize whose elements are all
labels = Variable(xp.array([i for i in range(10)]))  # make label

with chainer.using_config('train', False):
    x = gen(z, labels, 10)
x = chainer.cuda.to_cpu(x.data)  # send data to cpu

x = x * 127.5 + 127.5
x = x.transpose(0, 2, 3, 1)  # NCHW->NHWCに変形
x = combine_images(x)
plt.imshow(x, cmap=plt.cm.gray)
plt.axis("off")
plt.savefig("input_zeros.png")
plt.show()
