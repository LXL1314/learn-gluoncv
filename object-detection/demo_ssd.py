from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

net = model_zoo.ssd_512_resnet50_v1_voc(pretrained=True)

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='../img/street_small.jpg')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

class_IDs, scores, bounding_boxes = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDs[0], thresh=0.9, class_names=net.classes)
plt.show()