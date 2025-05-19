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


def draw(img, c=(0, 255, 0), thickness=20):
    """Draw a colored (usually red or green) box around an image."""
    p = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    for i in range(3):
        cv2.line(img, (p[i, 0], p[i, 1]), (p[i + 1, 0], p[i + 1, 1]), c, thickness=thickness * 2)
    return cv2.line(img, (p[3, 0], p[3, 1]), (p[0, 0], p[0, 1]), c, thickness=thickness * 2)


def build_prediction_image(img, images_paths, corrects):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    """
    labels = ["Query"]
    for i,_ in enumerate(images_paths):
        labels.append(f"Pred - {i + 1} \n Dist - {corrects[i][1]:.2f} ")

    num_images = len(images_paths) + 1
    images = [np.array(Image.open(path).convert("RGB")) for path in images_paths]

    for image, correct in zip(images, corrects):
        if correct is None:
            continue
        color = (0, 255, 0) if correct[0] else (255, 0, 0)
        draw(image, color)

    images.insert(0, np.array(img))

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

def save_preds(predictions, correct, eval_ds, log_dir, img, val):
    try:
        viz_dir = log_dir / "preds"
        viz_dir.mkdir()
    except: 
        pass 
        
    for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving preds in {viz_dir}/{val:05d}")):
        
        list_of_images_paths = []

        # List of None (query), True (correct preds) or False (wrong preds)
        for pred_index, pred in enumerate(preds):
            pred_path = str(eval_ds.dataset_path / eval_ds.image_paths[pred])
            list_of_images_paths.append(pred_path)

    prediction_image = build_prediction_image(img, list_of_images_paths, correct)

    pred_image_path = viz_dir / f"{val:05d}.jpg"

    prediction_image.save(pred_image_path)

    return prediction_image