import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from .grasp import GraspRectangles, detect_grasps


def plot_output(fig, rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img:
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.pause(0.1)
    fig.canvas.draw()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of network (Nx300x300x3)
    :param grasp_angle: Angle outputs of network
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from network
    :param threshold: Threshold for IOU matching. Detect with IOU ≥ threshold
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    else:
        return False
    
def acc(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, angle_threshold=np.pi/2, dimension_threshold=7.0):
    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        iou_percentage =  g.max_iou(gt_bbs)
            
    ##############################Berechnung des Winkelunterschieds#####################################
    angle_diff = abs((gt_bbs.angle - grasp_angle + np.pi / 2) % np.pi - np.pi / 2)
    if angle_diff > angle_threshold:
        angle_percentage = 0
    else:
        angle_percentage =  round(max(0, (1 - angle_diff / angle_threshold) * 100), 2)
    #print("Winkelunterschied in Grad:", round(np.degrees(angle_diff),4))
    print("Winkel Prozentsatz:", angle_percentage)

    ##############################Berechnung übereinstimmung Länge/Breite Prozentsatzes##########################
    ground_width = np.linalg.norm(ground_truth_bbs.points[0] - ground_truth_bbs.points[1])
    pred_width = grasp_width

    width_diff = abs(ground_width - pred_width)
    
    if width_diff >= dimension_threshold:
        dimension_percentage = 0
        print("Dimension Prozentsatz", dimension_percentage)
    
    else:
        width_percentage = max(0, (1 - width_diff / dimension_threshold) * 100)

        dimension_percentage= round((width_percentage) / 2, 2)
        print("Dimension Prozentsatz:", dimension_percentage)

    
    ##############################Durchschnitt der drei Prozentsätze############################
    average_percentage = round((angle_percentage + iou_percentage + dimension_percentage) / 3, 2)   
    
    return average_percentage
