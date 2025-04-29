import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import rescale
from tqdm import tqdm

# Height and width of a single image
H = 512
W = 512
TEXT_H = 175
FONTSIZE = 50
SPACE = 50  # Space between two images


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with text"""
    img = Image.new("RGB", ((W * len(labels)) + 50 * (len(labels) - 1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        _, _, w, h = d.textbbox((0, 0), text)
        d.text(((W + SPACE) * i + W // 2 - w // 2, 1), text, fill=(0, 0, 0))
    return np.array(img)[:100]  # Remove some empty space




def build_prediction_image(img, images_paths):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    """
    labels = ["Query"]
    for i,_ in enumerate(images_paths[1:]):
        labels.append(f"Pred - {i + 1}")

    num_images = len(images_paths) + 1
    images = [np.array(Image.open(path).convert("RGB")) for path in images_paths]
    images.insert(0, img)

    concat_image = np.ones([H, (num_images * W) + ((num_images - 1) * SPACE), 3])
    rescaleds = [
        rescale(i, [min(H / i.shape[0], W / i.shape[1]), min(H / i.shape[0], W / i.shape[1]), 1]) for i in images
    ]

    for i, image in enumerate(rescaleds):
        pad_width = (W - image.shape[1] + 1) // 2
        pad_height = (H - image.shape[0] + 1) // 2
        image = np.pad(image, [[pad_height, pad_height], [pad_width, pad_width], [0, 0]], constant_values=1)[:H, :W]
        concat_image[:, i * (W + SPACE) : i * (W + SPACE) + W] = image

    labels_image = write_labels_to_image(labels)
    final_image = np.concatenate([labels_image, concat_image])
    final_image = Image.fromarray((final_image * 255).astype(np.uint8))
    return final_image

def save_preds(predictions, eval_ds, log_dir, img, val):
    try:
        viz_dir = log_dir / "preds"
        viz_dir.mkdir()
    except: 
        pass 
        
    for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving preds in {viz_dir}")):
        
        list_of_images_paths = []

        # List of None (query), True (correct preds) or False (wrong preds)
        for pred_index, pred in enumerate(preds):
            pred_path = str(eval_ds.dataset_path / eval_ds.image_paths[pred])
            list_of_images_paths.append(pred_path)

    prediction_image = build_prediction_image(img, list_of_images_paths)

    pred_image_path = viz_dir / f"{val:03d}.jpg"

    prediction_image.save(pred_image_path)