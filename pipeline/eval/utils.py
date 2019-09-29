import matplotlib.pyplot as plt
import numpy as np


def draw_scores_scatterplot(list_score_name, list_list_score, gt_index, out_file):
    num_scores = len(list_score_name)

    fig, list_ax = plt.subplots(num_scores, 1)
    for score_name, list_score, ax in zip(list_score_name, list_list_score, list_ax):
        gt_score = list_score[gt_index]
        median_score = np.median(list_score)
        other_scores = list_score[:gt_index] + list_score[gt_index+1:]

        # Draw scatter plot.
        ax.scatter(other_scores, y=[0]*len(other_scores), c='blue', s=20, alpha=0.6)
        ax.scatter([median_score], y=[0], marker='|', c='green', s=170)  # median
        ax.scatter([gt_score], y=[0], c='red', s=35)  # GT viewpoint

        # Remove y-axes and set ylabel.
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_ylabel(score_name)

    plt.tight_layout()
    with open(out_file, 'wb') as f:
        plt.savefig(f, format='pdf', dpi=1000)
        plt.clf()

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou