# ROS

# pytorch
# import torch
# import torch.nn as nn
# from torchvision import transforms
# import torchvision.transforms.functional as TF
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage
from sensor_msgs.msg import Image

# from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


IMAGE_ASPECT_RATIO = 4 / 3


ACTION_STATS = {"min": [-2.5, -4], "max": [5, 4]}


def load_model_onnx(model_name: str):

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # providers = ["CUDAExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    ort_session = ort.InferenceSession(
        f"/workspace/src/deployment/model_weights/{model_name}",
        sess_options,
        providers=providers,
    )

    return ort_session


def load_model_trt(model_name: str):

    trt_model = TRTInfer(model_name)

    return trt_model


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes()
    ros_image.step = ros_image.width
    return ros_image


# def to_numpy(tensor):
#     return tensor.cpu().detach().numpy()


def transform_images(
    pil_imgs: list[PILImage.Image],
    image_size: list[int],
    center_crop: bool = False,
    return_img: bool = False,
):
    """Transforms a list of PIL image to a numpy"""

    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = center_crop_pil(
                    pil_img, (h, int(h * IMAGE_ASPECT_RATIO))
                )  # crop to the right ratio
            else:
                pil_img = center_crop_pil(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        if return_img:  # Added for debug purpose on rviz
            return pil_img
        transf_img = transform_numpy(pil_img)
        transf_img = np.expand_dims(transf_img, axis=0)
        transf_imgs.append(transf_img)
    return np.concatenate(transf_imgs, axis=1)


def transform_numpy(image):
    """
    Equivalent to:
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    Args:
        image (np.ndarray): Input image in shape (H, W, 3), values in [0, 255]
    Returns:
        np.ndarray: Normalized image in shape (3, H, W)
    """
    # Convert to float and scale to [0,1]
    # image = image.astype(np.float32) / 255.0

    # Change from HWC to CHW
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std

    return image


def center_crop_pil(img, output_size):
    """
    Args:
        img (PIL.Image): Input image
        output_size (tuple): (crop_height, crop_width)
    """
    w, h = img.size
    new_h, new_w = output_size

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))


# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
