from gluoncv import utils, data, model_zoo
from matplotlib import pyplot as plt

net = model_zoo.faster_rcnn_resnet50_v1b_voc(pretrained=True)
im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='../img/biking.jpg')
x, img = data.transforms.presets.rcnn.load_test(im_fname)
class_IDs, scores, bounding_boxes = net(x)

axe = utils.viz.plot_bbox(img, bboxes=bounding_boxes[0], scores=scores[0], labels=class_IDs[0], thresh=0.98, class_names=net.classes)
plt.show()

