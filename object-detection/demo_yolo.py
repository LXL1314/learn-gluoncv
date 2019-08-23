from gluoncv import utils, data, model_zoo
from matplotlib import pyplot as plt

net = model_zoo.yolo3_darknet53_voc(pretrained=True)
im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='../img/dog.jpg')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)

class_IDs, scores, bounding_boxes = net(x)
axe = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDs[0], thresh=0.9, class_names=net.classes)
plt.show()


