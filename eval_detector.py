import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    
    # Find intersection of the two boxes
    tl_row1, tl_col1, br_row1, br_col1 = box_1
    tl_row2, tl_col2, br_row2, br_col2 = box_2

    min_right = min(br_col1, br_col2)
    max_left = max(tl_col1, tl_col2)
    min_bottom = min(br_row1, br_row2)
    max_top = max(tl_row1, tl_row2)

    x_overlap = max(0, min_right - max_left + 1)
    y_overlap = max(0, min_bottom - max_top + 1)

    intersection = x_overlap * y_overlap

    box_1_area = (br_row1 - tl_row1 + 1) * (br_col1 - tl_col1 + 1)
    box_2_area = (br_row2 - tl_row2 + 1) * (br_col2 - tl_col2 + 1)
    union = box_1_area + box_2_area - intersection
    
    iou = intersection / union

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]

        for i in range(len(gt)):
            # Keep track of the number of predictions above a confidence value,
            # the number of true positives, and whether or not the ground truth
            # has a match yet
            num_preds = 0
            tp = 0
            found_match = False
            for j in range(len(pred)):
                if pred[j][4] >= conf_thr:
                    num_preds += 1
                    iou = compute_iou(pred[j][:4], gt[i])

                    # Match prediction with ground truth
                    if iou >= iou_thr and not found_match:
                        tp += 1
                        found_match = True

            # false positives are predictions with no matched ground truth
            # false negatives are ground truths with no matched prediction
            FP += num_preds - tp
            FN += len(gt) - tp
            TP += tp

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = './data/hw02_preds'
gts_path = './data/hw02_annotations'

# load splits:
split_path = './data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
iou_thrs = [0.25, 0.5, 0.75]

for iou_thr in iou_thrs:
    # Vary confidence thresholds on the training set for one IoU threshold.
    cvals = []
    for fname in preds_train:
        img_preds = preds_train[fname]
        for img_pred in img_preds:
            cvals.append(img_pred[4])

    confidence_thrs = np.sort(np.array(cvals ,dtype=float)) # using (ascending) list of confidence scores as thresholds
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr, conf_thr=conf_thr)

    num_preds = tp_train + fp_train
    num_gt = tp_train + fn_train

    # Compute precision and recall
    precision = np.divide(tp_train, num_preds, out=np.ones_like(tp_train), where=num_preds!=0)
    recall = np.divide(tp_train, num_gt, out=np.ones_like(tp_train), where=num_gt!=0)

    # Plot the precision and recall curves
    plt.figure(1)
    plt.plot(recall, precision, label=('iou = {}'.format(iou_thr)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve for training set, Weakened Alg.')
    plt.legend()

# Plot training set PR curves
plt.savefig('./data/plots/PR_curve_train_weak.jpg')
plt.show()

if done_tweaking:
    print('Code for plotting test set PR curves.')
    
    for iou_thr in iou_thrs:
        # Vary confidence thresholds on the training set for one IoU threshold.
        cvals = []

        for fname in preds_test:
            img_preds = preds_test[fname]
            for img_pred in img_preds:
                cvals.append(img_pred[4])

        confidence_thrs = np.sort(np.array(cvals ,dtype=float)) # using (ascending) list of confidence scores as thresholds
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr, conf_thr=conf_thr)

        num_preds = tp_test + fp_test
        num_gt = tp_test + fn_test

        # Compute precision and recall
        precision = np.divide(tp_test, num_preds, out=np.ones_like(tp_test), where=num_preds!=0)
        recall = np.divide(tp_test, num_gt, out=np.ones_like(tp_test), where=num_gt!=0)


        # Plot the precision and recall curves
        plt.figure(2)
        plt.plot(recall, precision, label=('iou = {}'.format(iou_thr)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve for testing set, Weakened Alg.')
        plt.legend()
    
    plt.savefig('./data/plots/PR_curve_test_weak.png')
    plt.show()

