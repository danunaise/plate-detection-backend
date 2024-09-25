import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation

def get_sidewalk_coords(image_path, show_plot=False):  # Added show_plot parameter
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    inputs = processor(images=image_np, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    logits = torch.nn.functional.interpolate(logits, size=image_np.shape[:2], mode='bilinear', align_corners=False)
    logits = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Define colors for each class
    colors = [
        (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50), (4, 200, 3),
        (120, 120, 80), (140, 140, 140), (204, 5, 255), (230, 230, 230), (4, 250, 7),
        (224, 5, 255), (235, 255, 7), (150, 5, 61), (120, 120, 70), (8, 255, 51),
        (255, 6, 82), (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
        (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255), (255, 7, 71),
        (255, 9, 224), (9, 7, 230), (220, 220, 220), (255, 9, 92), (112, 9, 255),
        (8, 255, 214), (7, 255, 224), (255, 184, 6), (10, 255, 71), (255, 41, 10),
        (7, 255, 255), (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
        (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255),
        (235, 12, 255), (160, 150, 20), (0, 163, 255), (140, 140, 140), (250, 10, 15),
        (20, 255, 0), (31, 255, 0), (255, 31, 0), (255, 224, 0), (153, 255, 0),
        (0, 0, 255), (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
        (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255), (0, 255, 112),
        (0, 255, 133), (255, 0, 0), (255, 163, 0), (255, 102, 0), (194, 255, 0),
        (0, 143, 255), (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173),
        (10, 0, 255), (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
        (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20), (255, 184, 184),
        (0, 31, 255), (0, 255, 61), (0, 71, 255), (255, 0, 204), (0, 255, 194),
        (0, 255, 82), (0, 10, 255), (0, 112, 255), (51, 0, 255), (0, 194, 255),
        (0, 122, 255), (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
        (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0), (8, 184, 170),
        (133, 0, 255), (0, 255, 92), (184, 0, 255), (255, 0, 31), (0, 184, 255),
        (0, 214, 255), (255, 0, 112), (92, 255, 0), (0, 224, 255), (112, 224, 255),
        (70, 184, 160), (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
        (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0), (255, 0, 235),
        (245, 0, 255), (255, 0, 122), (255, 245, 0), (10, 190, 212), (214, 255, 0),
        (0, 204, 255), (20, 0, 255), (255, 255, 0), (0, 153, 255), (0, 41, 255),
        (0, 255, 204), (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
        (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255), (184, 255, 0),
        (0, 133, 255), (255, 214, 0), (25, 194, 194), (102, 255, 0), (92, 0, 255),
    ]
    color_map = mcolors.ListedColormap(np.array(colors) / 255)

    sidewalk_color = (235, 255, 7)
    road_color = (140, 140, 140)

    sidewalk_index = np.where(np.all(np.array(color_map.colors) == np.array(sidewalk_color) / 255, axis=1))[0][0]
    road_index = np.where(np.all(np.array(color_map.colors) == np.array(road_color) / 255, axis=1))[0][0]

    sidewalk_mask = (logits == sidewalk_index)
    road_mask = (logits == road_index)

    dilated_sidewalk = binary_dilation(sidewalk_mask, iterations=3)
    dilated_road = binary_dilation(road_mask, iterations=3)

    intersection = dilated_sidewalk & dilated_road
    intersection_indices = np.argwhere(intersection)

    y_coords = intersection_indices[:, 0]
    x_coords = intersection_indices[:, 1]
    start_coords = (x_coords[np.argmin(y_coords)], np.min(y_coords))
    end_coords = (x_coords[np.argmax(y_coords)], np.max(y_coords))

    if show_plot:
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(logits, cmap=color_map, alpha=0.7, vmin=0, vmax=len(colors) - 1)
        plt.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 'r-', linewidth=2)
        plt.title("Segmentation Result with Path Boundary")
        plt.axis('off')
        plt.show()

    return start_coords, end_coords


# Example call within this module for testing
if __name__ == "__main__":
    start, end = get_sidewalk_coords('captured_frame.png', show_plot=True)
    print("Start:", start, "End:", end)
