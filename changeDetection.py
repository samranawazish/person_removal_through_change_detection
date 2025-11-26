import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def read_frames(path: str, ext: str = "png") -> np.ndarray:
    frames = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith(ext):
            img = Image.open(os.path.join(path, fname)).convert("RGB")
            frames.append(np.array(img))
    return np.array(frames)


def plot_frames(frames: np.ndarray, num_frames: int, save_name: str):
    cols = 5
    rows = (num_frames // cols) + 1
    plt.figure(figsize=(15, rows * 3))
    for i in range(min(num_frames, len(frames))):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(frames[i])
        plt.axis("off")
        plt.title(f"Frame {i}")
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
def compute_mean(frames: np.ndarray) -> np.ndarray:
    #mean using the formula μ = (Σx) / N
    sum_frame = np.zeros_like(frames[0], dtype=np.float64)
    for f in frames:
        sum_frame += f
    mean_frame = (sum_frame / len(frames)).astype(np.uint8)
    return mean_frame

def compute_variance(frames: np.ndarray, mean_frame: np.ndarray) -> np.ndarray:
    #variance using the formula σ² = (Σ(x-μ)²) / N
    sum_var = np.zeros_like(frames[0], dtype=np.float64)
    for f in frames:
        diff = f.astype(np.float64) - mean_frame.astype(np.float64)
        sum_var += diff ** 2
    variance_frame = (sum_var / len(frames)).astype(np.float64)
    return variance_frame

def compute_mask(frame, mean_frame, variance_frame, threshold=5.0):
    #Compute foreground mask using Mahalanobis-like distance
    diff = frame.astype(float) - mean_frame.astype(float)
    variance = variance_frame + 1e-6
    mdist = (diff ** 2) / variance
    mdist = np.sum(mdist, axis=2)
    mask = (mdist > threshold).astype(np.uint8) * 255
    return mask


#Morphological operations
def create_kernel(kernel_size=3):
    
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


def erode(mask, kernel, iterations=1):
    h, w = mask.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    out = mask.copy()
    for _ in range(iterations):
        temp = np.zeros_like(out)
        for i in range(ph, h - ph):
            for j in range(pw, w - pw):
                region = out[i - ph:i + ph + 1, j - pw:j + pw + 1]
                if np.all(region[kernel == 1] == 255):
                    temp[i, j] = 255
        out = temp
    return out


def dilate(mask, kernel, iterations=1):
    h, w = mask.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    out = mask.copy()
    for _ in range(iterations):
        temp = np.zeros_like(out)
        for i in range(ph, h - ph):
            for j in range(pw, w - pw):
                region = out[i - ph:i + ph + 1, j - pw:j + pw + 1]
                if np.any(region[kernel == 1] == 255):
                    temp[i, j] = 255
        out = temp
    return out

def morphological_operations(mask, kernel_size=3):
    kernel = create_kernel(kernel_size)
    eroded = erode(mask, kernel, iterations=1)
    dilated = dilate(eroded, kernel, iterations=1)
    return dilated


#Connected Components 
def find_connected_components(mask, connectivity=8):

    num_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity)
    return num_labels, labels


#Alpha Blending
def fade_into_background(foreground, background, binary_mask, alpha):
    binary_mask_3c = np.stack([binary_mask] * 3, axis=-1) // 255
    blended = (
        foreground * (1 - alpha * binary_mask_3c)
        + background * (binary_mask_3c * alpha)
    ).astype(np.uint8)
    return blended


#Save
def save_mask(mask, path, fname):
    img = Image.fromarray(mask)
    img.save(os.path.join(path, fname))

def changeDetection(input_folder, output_folder, input_ext, output_ext, video_format):
    os.makedirs(output_folder, exist_ok=True)

    frames = read_frames(input_folder, input_ext)
    plot_frames(frames, num_frames=20, save_name=os.path.join(output_folder, "frames_preview.pdf"))

    background_frames = frames[:70]
    mean_frame = compute_mean(background_frames)
    variance_frame = compute_variance(background_frames, mean_frame)
#videowriter
    h, w, _ = frames[0].shape
    video_path = os.path.join(output_folder, f"output.{video_format}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (w, h))

#processing frames
    for idx, frame in enumerate(frames):
        mask = compute_mask(frame, mean_frame, variance_frame, threshold=5.0)
        mask = morphological_operations(mask, kernel_size=3)
        num_labels, labels = find_connected_components(mask)
        alpha = min(1.0, idx / len(frames))
        blended = fade_into_background(frame, mean_frame, mask, alpha)
        fname = f"mask_{idx:04d}.{output_ext}"
        save_mask(mask, output_folder, fname)

        out.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Masks path {output_folder}")
    print(f"Video path {video_path}")
