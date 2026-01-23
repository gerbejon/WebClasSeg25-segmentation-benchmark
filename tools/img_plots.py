import numpy as np
# from ultralytics import YOLO
# import torch
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage import measure
import matplotlib.colors as mcolors  # For color conversion



def blend_segmentation_mask(img_mask, segm, names_dict=None, color_map=None, outfile=None, alpha=0.5, img = None, img_path=None):
    print(np.unique(img_mask))
    if color_map is None:
        color_map = {
            0: '#e6194b',  # red
            1: '#3cb44b',  # green
            2: '#ffe119',  # yellow
            3: '#4363d8',  # blue
            4: '#f58231',  # orange
            5: '#911eb4',  # purple
            6: '#46f0f0',  # cyan
            7: '#f032e6',  # magenta
            8: '#a52a2a',  # brown
            9: '#808080',  # dark gray
        }
    if names_dict is None:
        if segm == 'fc':
            names_dict = {1: 'header', 2: 'maincontent', 3: 'title', 4: 'image', 5: 'footer', 6: 'advertisement', 7: 'navigation', 0:'none'}
        elif segm == 'mc':
            names_dict = {0: "information1", 1: "information2", 2: "interaction", 3: "transaction1", 4: "transaction2", 5: "transaction3", 6: "integration1", 7: "integration2",  8:'none'}
        else:
            print('invalid segmentation method, returning...')
        # names_dict =  {"1": "header", "2": "maincontent", "3": "title", "4": "image", "5": "footer", "6": "advertisement", "7": "navigation", "0": "none"}
    # Load grayscale image and convert to RGBA
    if img is None:
        img = Image.open(img_path).convert('L').convert('RGBA')
    img_np = np.array(img.convert('L').convert('RGBA'))

    # Build a color RGBA overlay image
    seg_rgba = np.zeros((*img_mask.shape, 4), dtype=np.uint8)
    class_ids = np.unique(img_mask)

    for class_id in class_ids:
        if class_id == 9:
            continue  # skip background if needed

        rgba = np.array(mcolors.to_rgba(color_map[int(class_id )])) * 255
        rgba = rgba.astype(np.uint8)
        rgba[3] = int(alpha * 255)  # apply transparency

        seg_rgba[img_mask == class_id] = rgba

    # Convert mask to RGBA PIL image
    seg_img = Image.fromarray(seg_rgba, mode='RGBA')

    # Blend grayscale image with segmentation overlay
    blended = Image.alpha_composite(Image.fromarray(img_np), seg_img)

    # Plot the blended image
    dpi = 100
    height, width = img_mask.shape
    figsize = width / dpi, height / dpi
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(blended)
    ax.axis('off')

    # Draw contours and build legend
    legend_handles = []
    for class_id in class_ids:
        if class_id == 9 and segm == 'fc':
            continue
        elif class_id == 8 and segm == 'mc':
            continue
        binary_mask = (img_mask == class_id).astype(np.uint8)
        contours = measure.find_contours(binary_mask, level=0.5)

        # if segm == 'fc':
        #     color = color_map[int(class_id - 1)]
        # else:
        #     color = color_map[int(class_id) ]
        color = color_map[int(class_id)]
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5)

        legend_handles.append(Patch(color=color, label=names_dict[int(class_id)]))

    ax.legend(handles=legend_handles, loc='lower right', frameon=True)

    # Save result

    if outfile is None:
        img_id = img_path.split('/')[-1].split('.')[0]
        outfile = f"results/yolo_{img_id}_segmentation_contours_segment.png"
    if not os.path.exists('/'.join(outfile.split('/')[:-1])):
        os.makedirs('/'.join(outfile.split('/')[:-1]))

    plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
    plt.close(fig)