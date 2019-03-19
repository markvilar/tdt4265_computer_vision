import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from task2_tools import read_predicted_boxes, read_ground_truth_boxes

def calculate_box_area(box):
    """
    Calculates the area of a bounding box.

    Args:
        box (np.array of floats): bounding box
            [xmin, ymin, xmax, ymax]
    Returns:
        float: the area of the bounding box
    """
    del_x = box[2] - box[0]
    del_y = box[3] - box[1]
    return del_x * del_y


def is_overlapping(box1, box2):
    """
    Checks where two bounding boxes are overlapping.

    Args:
        box1 (np.array of floats): bounding box
            [xmin, ymin, xmax, ymax]
        box2 (np.array of floats): bounding box
            [xmin, ymin, xmax, ymax]
    Returns:
        boolean: whether the bounding boxes are overlapping or not
    """
    if box1[2] <= box2[0]: # If box1 is to the left of box2
        return False
    elif box1[0] >= box2[2]: # If box1 is to the right of box2
        return False
    elif box1[3] <= box2[1]: # If box1 is below box2
        return False
    elif box1[1] >= box2[3]: # If box1 is above box2
        return False
    else:
        return True


def get_overlap_box(box1, box2):
    """
    Calculates the bounding box of the overlap of box1 and box2.

    Args:
        box1 (np.array of floats): bounding box
            [xmin, ymin, xmax, ymax]
        box2 (np.array of floats): bounding box
            [xmin, ymin, xmax, ymax]
    Returns:
        np.array of floats: the bounding box defining the overlap of box1 and box2
            [xmin, ymin, xmax, ymax]
    """
    xmax = np.minimum(box1[2], box2[2])
    xmin = np.maximum(box1[0], box2[0])
    ymax = np.minimum(box1[3], box2[3])
    ymin = np.maximum(box1[1], box2[1])

    return np.array([xmin, ymin, xmax, ymax])


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the intersection of union for the two boxes.
    """
    if is_overlapping(prediction_box, gt_box):
        overlap_box = get_overlap_box(prediction_box, gt_box)

        intersection = calculate_box_area(overlap_box)
        pred_box_area = calculate_box_area(prediction_box)
        gt_box_area = calculate_box_area(gt_box)
        
        union = pred_box_area + gt_box_area - intersection
        return intersection / union  
    else:
        return 0


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0
    else:
        return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Return empty arrays if the arguments are invalid
    if prediction_boxes.size == 0 or gt_boxes.size == 0:
        return np.array([]), np.array([])
    elif prediction_boxes.ndim != 2:
        return np.array([]), np.array([])
    elif gt_boxes.ndim != 2:
        return np.array([]), np.array([])
    elif prediction_boxes.shape[1] != 4:
        return np.array([]), np.array([])
    elif gt_boxes.shape[1] != 4:
        return np.array([]), np.array([])
    
    n_points = prediction_boxes.shape[1] 
    matches = []
    best_matches = []

    # Find all possible matches with a IoU >= iou threshold
    for i, prediction_box in enumerate(prediction_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(prediction_box, gt_box)

            if iou >= iou_threshold:
                matches.append([i, j, iou])
    
    # Return empty arrays if there are no matches
    if len(matches) == 0:
        return np.array([]), np.array([])

    # Sort all matches based on IoU in descending order
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # Find all matches with the highest IoU threshold
    for match in matches:
        # Appends the indices of the prediction box and the ground truth box if
        # neither of them exists in best_matches
        if not any(match[0] ==  best_match[0] for best_match in best_matches):
            if not any(match[1] == best_match[1] for best_match in best_matches):
                best_matches.append(match)
    
    matched_prediction_boxes = np.zeros(shape=(len(matches), n_points))
    matched_gt_boxes = np.zeros(shape=(len(matches), n_points))

    for i, best_match in enumerate(best_matches):
        matched_prediction_boxes[i] = prediction_boxes[best_match[0]]
        matched_gt_boxes[i] = gt_boxes[best_match[1]]
    
    return matched_prediction_boxes, matched_gt_boxes


def calculate_individual_image_result(
        prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    # Find the bounding box matches with the highes IoU threshold
    matched_prediction_boxes, matched_gt_boxes = get_all_box_matches(
            prediction_boxes,
            gt_boxes,
            iou_threshold)
    
    # Compute true positives, false positives, false negatives
    num_tp = len(matched_prediction_boxes)
    num_fp = len(prediction_boxes) - num_tp
    num_fn = np.maximum(len(gt_boxes) - len(matched_gt_boxes), 0, dtype=int)

    return {"true_pos": num_tp, "false_pos": num_fp, "false_neg": num_fn}


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images.
       
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Find total true positives, false positives and false negatives
    # over all images

    tot_num_tp, tot_num_fp, tot_num_fn = 0, 0, 0

    for image in range(len(all_prediction_boxes)):
        prediction_boxes = all_prediction_boxes[image]
        gt_boxes = all_gt_boxes[image]

        image_results = calculate_individual_image_result(
                prediction_boxes,
                gt_boxes,
                iou_threshold)

        tot_num_tp += image_results["true_pos"]
        tot_num_fp += image_results["false_pos"]
        tot_num_fn += image_results["false_neg"]

    # Compute precision, recall
    precision = calculate_precision(tot_num_tp, tot_num_fp, tot_num_fn)
    recall = calculate_recall(tot_num_tp, tot_num_fp, tot_num_fn)

    return (precision, recall)


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the precision-recall curve over all images. Use the given
       confidence thresholds to find the precision-recall curve.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the 
        list is a np.array containing all predicted bounding boxes for the 
        given image.
            Shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the 
            given image objects
            Shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        confidence_scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. 
            Shape: [number of predicted boxes]

            E.g: confidence_score[0][1] is the confidence score for a predicted 
            bounding 
            box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both np.array of floats.
    """
    # Instead of going over every possible confidence score threshold to 
    # compute the PR curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run
    # the final evaluation
    confidence_thresholds = np.linspace(0, 1, 500)

    precisions = np.zeros(len(confidence_thresholds))
    recalls = np.zeros(len(confidence_thresholds))
    
    for i, conf_threshold in enumerate(confidence_thresholds):
        
        filtered_prediction_boxes = []
        
        for image in range(0, len(confidence_scores)):
            img_conf_scores = confidence_scores[image]
            img_pred_boxes = all_prediction_boxes[image]
            img_valid_idxs = np.argwhere(img_conf_scores >= conf_threshold)
            img_valid_pred_boxes = img_pred_boxes[img_valid_idxs[:,0]]
            filtered_prediction_boxes.append(img_valid_pred_boxes)
        
        precision, recall = calculate_precision_recall_all_images(
                filtered_prediction_boxes,
                all_gt_boxes,
                iou_threshold)
        
        precisions[i] = precision
        recalls[i] = recall

    return (precisions, recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run 
    # the final evaluation
    recall_levels = np.linspace(0, 1.0, 11)
    interpolated_precisions = np.zeros(shape=recall_levels.shape)

    # YOUR CODE HERE
    for i, recall_level in enumerate(recall_levels):
        valid_idxs = np.argwhere(recalls >= recall_level)
        filtered_precisions = precisions[valid_idxs[:,0]]

        if len(filtered_precisions) > 0:
            interpolated_precisions[i] = np.amax(filtered_precisions)
    
    mean_average_precision = np.mean(interpolated_precisions)

    return mean_average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
