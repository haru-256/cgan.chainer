import chainer
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    def __init__(self, bottom_width=3, ch=11, wscale=0.02):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # initializers
            w = chainer.initializers.Normal(wscale)

            # register layer with variable
            self.c0 = L.Convolution2D(
                in_channels=ch,
                out_channels=64,
                ksize=5,
                stride=2,
                pad=2,
                initialW=w)
            self.c1 = L.Convolution2D(
                in_channels=None,
                out_channels=32,
                ksize=3,
                stride=2,
                pad=1,
                initialW=w)
            self.c2 = L.Convolution2D(
                in_channels=None,
                out_channels=16,
                ksize=3,
                stride=1,
                pad=1,
                initialW=w)

            self.l4 = L.Linear(in_size=None, out_size=1, initialW=w)

            # self.bn0 = L.BatchNormalization(size=ch//8, use_gamma=False)
            self.bn1 = L.BatchNormalization(size=32, use_gamma=False)
            self.bn2 = L.BatchNormalization(size=16, use_gamma=False)
            self.bn3 = L.BatchNormalization(size=8, use_gamma=False)

    def __call__(self, x, labels, num_labels):
        """
        Function that computes forward

        Parametors
        ----------------
        x: Variable
           input image data. this shape is (N, C, H, W)
    
        labels: Variable
           labels data. this shaoe is (N, num_label)

        num_labels: int
           number of labels
        """
        one_hot_labels = self.to_one_hot(labels, num_labels, x.data.shape[0])
        h = F.concat((x, one_hot_labels), axis=1)
        h = F.leaky_relu(self.c0(h))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        y = self.l4(h)  # conv->linear では勝手にreshapeが適用される

        return y

    def to_one_hot(self, labels, num_labels, num_data):
        """
        make one_hot labels whose shape is (N, C, H, W)

        Parametors
        --------------
        labels: Variable
           labels data. this shaoe is (N, num_label)

        num_labels: int
           number of labels

        num_data: int
           number of datas
        """
        xp = chainer.cuda.get_array_module(labels.data)
        one_hot_labels = xp.eye(num_labels)[labels.data].astype("f")
        one_hot_labels = one_hot_labels.reshape(num_data, num_labels, 1, 1)
        mask = xp.ones((num_data, num_labels, 28, 28), dtype="f")

        return one_hot_labels * mask


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable
    import numpy as np

    z = np.random.uniform(-1, 1, (1, 1, 28, 28)).astype("f")
    labels = np.array([1])
    model = Discriminator()
    img = model(Variable(z), Variable(labels), 10)
    # print(img)
    g = c.build_computational_graph(img)
    with open('dis_graph.dot', 'w') as o:
        o.write(g.dump())
