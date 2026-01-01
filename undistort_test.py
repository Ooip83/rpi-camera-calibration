import cv2
import pickle
import matplotlib.pyplot as plt

# ===== USER SETTINGS =====
pickle_file = "dist_pickle.p"     # contains mtx and dist
image_path  = "Calibration Image Set/T15.jpg"   # one distorted image
# ========================

# Load calibration data
with open(pickle_file, "rb") as f:
    data = pickle.load(f)

mtx  = data["mtx"]
dist = data["dist"]

# Load image
img = cv2.imread(image_path)
img = cv2.resize(img, (640, 320))
if img is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

h, w = img.shape[:2]

# Undistort
undistorted = cv2.undistort(img, mtx, dist, None, mtx)

# Convert BGR → RGB for matplotlib
img_rgb        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
undist_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Undistorted")
plt.imshow(undist_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()
