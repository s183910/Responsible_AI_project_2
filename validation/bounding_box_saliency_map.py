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


# set up model, pretrained on imagenet
model = resnet34(weights='DEFAULT').eval()
# don't really know what this does (especially the normalize thing - were do the variables come from?
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# to visualize the images in same format
preprocess_visualise = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor()
])

# loading the test images
imgs = glob.glob('handful_of_images/*.jpg')
len(imgs)

cam_extractor = SmoothGradCAMpp(model,target_layer='layer4')
def get_heat_map(fileNames, overlay):
    #getting the saliency maps on the images
    fig, ax = plt.subplots(2,5, tight_layout=True)
    ax = ax.ravel()
    for i, img in enumerate(fileNames):
        input_image = Image.open(img)
        cropped_img = preprocess_visualise(input_image)

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        out = model(input_batch)
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)

        if overlay:
            result = overlay_mask(to_pil_image(cropped_img), to_pil_image(cams[0].squeeze(0), mode='F'), alpha=0.5)
            ax[i].imshow(result); ax[i].set_axis_off();
        else:
            ax[i].imshow(cams[0].squeeze(0)); ax[i].set_axis_off();
    plt.show()


get_heat_map(imgs, False)

cam_extractor.remove_hooks()



