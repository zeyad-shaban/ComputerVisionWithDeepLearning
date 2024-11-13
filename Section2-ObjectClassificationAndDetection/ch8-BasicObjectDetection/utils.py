import torch
from matplotlib.patches import Rectangle
from selectivesearch import selective_search
import matplotlib.pyplot as plt

def plot_box(sample_box: torch.Tensor):
    assert(sample_box.shape[0] == 4)
    plt.gca().add_patch(Rectangle((sample_box[0], sample_box[1]), sample_box[2] - sample_box[0], sample_box[3] - sample_box[1], facecolor='none', edgecolor='r'))

def get_candidates(img: torch.Tensor) -> torch.Tensor:
    _, regions = selective_search(img.int().permute(1, 2, 0), scale=0.0000001, min_size=0)
    img_area = img.shape[2] * img.shape[1]
    rects = [region['rect'] for region in regions if region['size'] > img_area * 0.05]
    return torch.tensor(rects)


def iou(boxA, boxB, epsilon=1e-8):
    """
    Computes the IoU between multiple pairs of boxes.

    Parameters:
    boxA (Tensor[N, 4]): Tensor containing N boxes in format [x1, y1, x2, y2]
    boxB (Tensor[M, 4]): Tensor containing M boxes in format [x1, y1, x2, y2]
    epsilon (float): Small value to avoid division by zero

    Returns:
    Tensor[N, M]: IoU values for each pair of boxes from boxA and boxB
    """

    # Find the coordinates of the intersection boxes
    x1_inter = torch.max(boxA[:, None, 0], boxB[:, 0])  # Shape [N, M]
    y1_inter = torch.max(boxA[:, None, 1], boxB[:, 1])
    x2_inter = torch.min(boxA[:, None, 2], boxB[:, 2])
    y2_inter = torch.min(boxA[:, None, 3], boxB[:, 3])

    # Compute the area of the intersection
    inter_area = torch.clamp((x2_inter - x1_inter), min=0) * torch.clamp((y2_inter - y1_inter), min=0)

    # Compute the area of both sets of boxes
    boxA_area = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])  # Shape [N]
    boxB_area = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])  # Shape [M]

    # Compute the union area
    union_area = boxA_area[:, None] + boxB_area - inter_area  # Shape [N, M]

    # Compute IoU
    iou = inter_area / (union_area + epsilon)  # Shape [N, M]

    return iou

boxA = torch.tensor([[50, 50, 150, 150], [30, 30, 100, 100]])  # Shape [2, 4]
boxB = torch.tensor([[60, 60, 160, 160], [20, 20, 80, 80], [50, 50, 120, 120]])  # Shape [3, 4]

iou_matrix = iou(boxA, boxB)
print(iou_matrix)
