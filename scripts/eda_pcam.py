import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

data_dir = "surfdrive"
train_x_file = os.path.join(data_dir, "camelyonpatch_level_2_split_train_x.h5")
train_y_file = os.path.join(data_dir, "camelyonpatch_level_2_split_train_y.h5")
val_x_file = os.path.join(data_dir, "camelyonpatch_level_2_split_valid_x.h5")
val_y_file = os.path.join(data_dir, "camelyonpatch_level_2_split_valid_y.h5")

plots_dir = "scripts/plots"
os.makedirs(plots_dir, exist_ok=True)

with h5py.File(train_x_file, "r") as f_x, h5py.File(train_y_file, "r") as f_y:
    X_train = f_x["x"][:]
    y_train = f_y["y"][:]

with h5py.File(val_x_file, "r") as f_x, h5py.File(val_y_file, "r") as f_y:
    X_val = f_x["x"][:]
    y_val = f_y["y"][:]

print(f"Train images shape: {X_train.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Validation images shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

y_train_flat = y_train.reshape(-1)
y_val_flat = y_val.reshape(-1)

plt.figure(figsize=(6, 4))
classes, counts = np.unique(y_train_flat, return_counts=True)
plt.bar(classes, counts, color=["skyblue", "salmon"])
plt.xticks(classes)
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("PCAM Train Class Distribution")
plt.savefig(os.path.join(plots_dir, "train_class_distribution.png"))
plt.close()

for cls in [0, 1]:
    cls_idx = np.where(y_train_flat == cls)[0]
    sample_idx = cls_idx[:5]
    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(sample_idx):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_train[idx])
        plt.axis("off")
        plt.title(f"Class {cls}")
    plt.suptitle(f"Sample Images - Class {cls}")
    plt.savefig(os.path.join(plots_dir, f"sample_images_class_{cls}.png"))
    plt.close()

mean_intensity = X_train.mean(axis=(1, 2, 3))
plt.figure(figsize=(6, 4))
plt.hist(mean_intensity, bins=50, color="lightgreen")
plt.xlabel("Mean pixel intensity")
plt.ylabel("Number of images")
plt.title("Histogram of Mean Pixel Intensity (Train Set)")
plt.savefig(os.path.join(plots_dir, "mean_pixel_intensity.png"))
plt.close()

print(f"Plots saved to '{plots_dir}' folder.")
