import cv2
import matplotlib.pyplot as plt

# Load the image (grayscale for SIFT)
image_path = 'img.png'
og = cv2.imread(image_path)
image = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

print(f"Number of keypoints detected: {len(keypoints)}")
print(f"Descriptor shape: {descriptors.shape}")  # (num_keypoints, 128)

# Draw keypoints on the image
sift_image = cv2.drawKeypoints(og, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Convert BGR to RGB for visualization in matplotlib
sift_image_rgb = cv2.cvtColor(sift_image, cv2.COLOR_BGR2RGB)

# Display the image with keypoints
plt.imshow(sift_image_rgb)
plt.title("SIFT Features")
plt.axis('off')
plt.savefig('sift.png', dpi=300, bbox_inches='tight', transparent=True)
