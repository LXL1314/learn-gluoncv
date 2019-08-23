import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model

url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/plane-draw.jpeg'
im_fname = utils.download(url, path='../img/plane-draw.jpeg')
img = image.imread(im_fname)
plt.imshow(img.asnumpy())
plt.show()

transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
img = transform_fn(img)
plt.imshow(nd.transpose(img, (1, 2, 0)).asnumpy())
plt.show()

ctx = mx.gpu()
net = get_model('cifar_resnet110_v1', classes=10, pretrained=True, ctx=ctx)
pred = net(img.copyto(ctx).expand_dims(axis=0))
print("pred: ", pred)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

ind = nd.argmax(pred, axis=1).astype('int').asscalar()

print("the picture class: %s, probability: %.3f" % (class_names[ind], nd.softmax(pred)[0][ind].asscalar()))
