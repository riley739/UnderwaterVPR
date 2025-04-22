import cv2
import numpy as np
import os

def enhance_underwater_image(img):
    # White Balance
    img = white_balance(img)

    # CLAHE in LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def white_balance(img):
    result = img.copy().astype(np.float32)
    avg_b, avg_g, avg_r = [np.mean(result[:, :, i]) for i in range(3)]
    avg_gray = (avg_b + avg_g + avg_r) / 3
    for i, avg in enumerate([avg_b, avg_g, avg_r]):
        result[:, :, i] *= (avg_gray / avg)
    return np.clip(result, 0, 255).astype(np.uint8)

def process_folder(folder_path):
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(supported):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {filename}")
                continue
            
            enhanced = enhance_underwater_image(img)
            combined = np.hstack((cv2.resize(img, (600, 400)), cv2.resize(enhanced, (600, 400))))

            cv2.imshow("Original (Left) vs Enhanced (Right)", combined)
            key = cv2.waitKey(0)
            if key == 27:  # ESC key
                break
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    folder_path = "path/to/your/images"
    process_folder(folder_path)
