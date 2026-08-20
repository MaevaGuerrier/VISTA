import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from torch import nn
from torchvision import transforms
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO
from vint_train.models.vint.vint import ViNT


bridge = CvBridge()


def pil_to_numpy_array(image_input, target_size: tuple = (224, 224)) -> np.ndarray:
    """Convert PIL image or numpy array to numpy array with proper formatting for Crossformer."""

    if isinstance(image_input, PILImage.Image):
        if image_input.size != target_size:
            print(f"Resizing image from {image_input.size} to {target_size} PIL")
            image_input = image_input.resize(target_size)
        img_array = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        print(f"Resizing image from {image_input.size} to {target_size} NDARRAY")

        img_array = image_input.copy()

        if img_array.shape[:2] != target_size:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))
            elif len(img_array.shape) == 2:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8), mode="L")
            else:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))

            pil_temp = pil_temp.resize(target_size)
            img_array = np.array(pil_temp)
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    return img_array


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:

    model_type = config["model_type"]

    if model_type == "vint_dino":
        model = ViNTWithDINOTokens(
            image_size=config["image_size"],
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            encoding_size=config["obs_encoding_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            output_layers=config["output_layers"],
            positional_encoding_type=config.get("positional_encoding_type", "peg"),
            separate_tokens_and_heads=config.get("separate_tokens_and_heads", False),
            take_action_history=config.get("take_action_history", False),
            action_enc_layers=config.get("action_enc_layers", [256]),
        )
    elif model_type == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    # elif model_type == "vint_da":
    #     model = ViNTWithDepthAnything(
    #         image_size=config["image_size"],
    #         context_size=config["context_size"],
    #         len_traj_pred=config["len_traj_pred"],
    #         learn_angle=config["learn_angle"],
    #         obs_encoder=config["obs_encoder"],
    #         encoding_size=config["obs_encoding_size"],
    #         mha_num_attention_heads=config["mha_num_attention_heads"],
    #         mha_num_attention_layers=config["mha_num_attention_layers"],
    #         mha_ff_dim_factor=config["mha_ff_dim_factor"],
    #         output_layers=config["output_layers"],
    #         positional_encoding_type=config.get("positional_encoding_type", "peg"),
    #         separate_tokens_and_heads=config.get("separate_tokens_and_heads", False),
    #         add_temporal_pe=config.get("add_temporal_pe", False),
    #     )
    else:
        raise ValueError(f"Model {config['model_type']} not supported")

    checkpoint = torch.load(model_path, map_location=device)

    loaded_model = checkpoint["model"]
    try:
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict, strict=True)
    except AttributeError:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


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


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(
    pil_imgs: list[PILImage.Image],
    image_size: list[int],
    center_crop: bool = False,
    return_img: bool = False,
) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(
                    pil_img, (h, int(h * IMAGE_ASPECT_RATIO))
                )  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        if return_img:  # Added for debug purpose on rviz
            return pil_img
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


# # FUNCTIONS FOR IMAGE OVERLAY WITH TRAJECTORIES


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    Args:
        xy: array of shape (batch_size, horizon, 2) representing (x, y) coordinates
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = np.zeros((3, 1), dtype=np.float64)

    xyz[..., 0] += camera_x_offset

    # Convert from (x, y, z) to (y, -z, x) for cv2
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)

    # done for cv2.fisheye.projectPoint requires float32/float64 and shape (N,1,3),
    xyz_cv = xyz_cv.reshape(batch_size * horizon, 1, 3).astype(np.float64)

    # uv, _ = cv2.projectPoints(
    #     xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    # )
    uv, _ = cv2.fisheye.projectPoints(xyz_cv, rvec, tvec, camera_matrix, dist_coeffs)

    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    viz_img_size: tuple[int, int],
):
    """
    Projects 3D coordinates onto a 2D image plane using the provided camera parameters.
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    # print(pixels)
    # Flip image horizontally
    pixels[:, 0] = viz_img_size[0] - pixels[:, 0]

    return pixels


def pil_to_numpy_array(image_input, target_size: tuple = (224, 224)) -> np.ndarray:
    """Convert PIL image or numpy array to numpy array with proper formatting for Crossformer."""

    if isinstance(image_input, PILImage.Image):
        if image_input.size != target_size:
            print(f"Resizing image from {image_input.size} to {target_size} PIL")
            image_input = image_input.resize(target_size)
        img_array = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        print(f"Resizing image from {image_input.size} to {target_size} NDARRAY")

        img_array = image_input.copy()

        if img_array.shape[:2] != target_size:
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))
            elif len(img_array.shape) == 2:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8), mode="L")
            else:
                pil_temp = PILImage.fromarray(img_array.astype(np.uint8))

            pil_temp = pil_temp.resize(target_size)
            img_array = np.array(pil_temp)
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    return img_array


def plot_trajs_and_points_on_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    list_trajs: list,
    viz_img_size: tuple[int, int],
    camera_height: float,
    camera_x_offset: float,
    resize_factor: bool = False,
):
    """
    Plot trajectories and points on an image.
    resize_factor: if True resize the image to viz_img_size. This is needed due to the fact that orginal image coming from fisheye is 640 x 480 and the traversability image is 224 x 224.
    Thus the camera matrix needs to be scaled accordingly.
    """

    for traj in list_trajs:
        xy_coords = traj[:, :2]
        traj_pixels = get_pos_pixels(
            xy_coords,
            camera_height,
            camera_x_offset,
            camera_matrix,
            dist_coeffs,
            viz_img_size,
        )

        if resize_factor:  # Traversability image is 224 x 224 and the original fisheye image is 640 x 480
            traj_pixels[:, 0] *= 0.35
            traj_pixels[:, 1] *= 0.46

        points = traj_pixels.astype(int).reshape(-1, 1, 2)

        color = tuple(int(x) for x in np.random.choice(range(50, 255), size=3))

        # inverting x,y axis so origin in image is down-left corner
        if resize_factor:
            points[:, :, 1] = viz_img_size[1] * 0.46 - 1 - points[:, :, 1]
        else:
            points[:, :, 1] = viz_img_size[1] - 1 - points[:, :, 1]

        # Draw trajectory
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=2)

    return img
