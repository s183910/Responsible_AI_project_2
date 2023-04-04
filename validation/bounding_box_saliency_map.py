import torch
from torchvision.models import resnet18, resnet34
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp
from torchvision.io import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp, LayerCAM

# https://frgfm.github.io/torch-cam/methods.html#torchcam.methods.SmoothGradCAMpp
# I believe this is smoothgrad from the lecture
from PIL import Image
from torchvision import transforms
import cv2
import glob
import operator

## import the necessary packages
from collections import namedtuple
import numpy as np
import cv2  # pip install opencv-python
from google.colab.patches import cv2_imshow


# set up model, pretrained on imagenet
model = resnet34(weights="DEFAULT").eval()
# don't really know what this does (especially the normalize thing - were do the variables come from?
preprocess = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# to visualize the images in same format
preprocess_visualise = transforms.Compose(
    [transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()]
)

# loading the test images
imgs = glob.glob("handful_of_images/*.jpg")
len(imgs)

cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")


def get_heat_map(fileNames, overlay):
    # getting the saliency maps on the images
    fig, ax = plt.subplots(2, 5, tight_layout=True)
    ax = ax.ravel()
    for i, img in enumerate(fileNames):
        input_image = Image.open(img)
        cropped_img = preprocess_visualise(input_image)

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        out = model(input_batch)
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)

        if overlay:
            result = overlay_mask(
                to_pil_image(cropped_img), to_pil_image(cams[0].squeeze(0), mode="F"), alpha=0.5
            )
            ax[i].imshow(result)
            ax[i].set_axis_off()
        else:
            ax[i].imshow(cams[0].squeeze(0))
            ax[i].set_axis_off()

    plt.show()


# get_heat_map(imgs, False)

cam_extractor.remove_hooks()

# ## define the `Detection` object
# Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


# def bb_intersection_over_union(boxA, boxB):
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     # compute the area of intersection rectangle
#     interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#     boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     # return the intersection over union value
#     return iou


# def IoU_saliency(fileNames, id, overlay):
#     ax = ax.ravel()
#     for i, img in enumerate(fileNames):
#         input_image = Image.open(img)
#         cropped_img = preprocess_visualise(input_image)

#         input_tensor = preprocess(input_image)
#         input_batch = input_tensor.unsqueeze(0)

#         out = model(input_batch)
#         cams = cam_extractor(out.squeeze(0).argmax().item(), out)

#         val = []
#         for i in range(0, cams[0].squeeze(0).shape[0]):
#             index, value = max(enumerate(cams[0].squeeze(0)[i]), key=operator.itemgetter(1))
#             val.append(value)

#         y_index, y_value = max(enumerate(val), key=operator.itemgetter(1))
#         x_index, x_value = max(enumerate(cams[0].squeeze(0)[y_index]), key=operator.itemgetter(1))

#         imz = Image.open(imgs).convert("F")
#         wid, hei = imz.size

#         cms = cams[0].squeeze(0).shape[0]

#         x_ = wid // cms
#         y_ = hei // cms
#         x = (x_index) * x_
#         y = (y_index) * y_

#         examples = [
#             Detection(
#                 "handful_of_images/Pigeon_Guillemot_0109_39872.jpg",
#                 [71, 15, 71 + 139, 181 + 15],
#                 [(x - (wid // cms)), (y - (hei // cms)), (x + (wid // cms)), (y + (hei // cms))],
#             )
#         ]

#         # loop over the example detections
#         for detection in examples:
#             # load the image
#             image = cv2.imread(detection.image_path)
#             # draw the ground-truth bounding box along with the predicted
#             # bounding box
#             cv2.rectangle(image, tuple(detection.gt[:2]), tuple(detection.gt[2:]), (0, 255, 0), 2)
#             cv2.rectangle(
#                 image, tuple(detection.pred[:2]), tuple(detection.pred[2:]), (0, 0, 255), 2
#             )
#             # compute the intersection over union and display it
#             iou = bb_intersection_over_union(detection.gt, detection.pred)
#             cv2.putText(
#                 image,
#                 "IoU: {:.4f}".format(iou),
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0, 255, 0),
#                 2,
#             )
#             print("{}: {:.4f}".format(detection.image_path, iou))
#             # show the output image
#             cv2_imshow(image)
#             cv2.waitKey(0)


# IoU_saliency(imgs, 0, False)

# cam_extractor.remove_hooks()
