import numpy as np



def compute_conformity_scores(scores, l=0., kreg=0.):
    ''' 
    Compute conformity score of every class for each image/bbox.

    Arguments:
    - scores [np.array]: predicted scores for every image/bbox (N_images, N_classes)
    - l, kreg [float]: parameters used to regularise the conformity score, as introduced in 'https://arxiv.org/pdf/2009.14193.pdf'

    Ouputs:
    - conformity_score [np.array]: conformity scores for every image/bbox, sorted in decreasing order
    - sorted_idxs [np.array]: class indices corresponding to the sorted conformity scores.
    '''
    sorted_idxs = np.argsort(-scores)
    sorted_scores = []
    for s, sid in zip (scores, sorted_idxs):
        sorted_scores.append(s[sid])
    sorted_scores = np.stack(sorted_scores)
        
    conformity_score = np.cumsum(sorted_scores, axis=1)
    
    reg = np.clip( l * (np.arange(1,scores.shape[1]+1)-kreg), 0, None )
    
    conformity_score += reg

    return conformity_score, sorted_idxs



def calibrate_cp_threshold(scores, gt_labels, alpha, l=0., kreg=0.)
    '''
    Calibrate the conformity-score threshold for a given significance level.

    Arguments:
    - scores [np.array]: predicted scores for every image/bbox (N_images, N_classes) in the calibration set
    - gt_labels [np.array]: ground-truth class id of every images/bboxes (N_images)
    - alpha [float]: significance level
    - l, kreg [float]: parameters used to regularise the conformity score, as introduced in 'https://arxiv.org/pdf/2009.14193.pdf'

    Output:
    - cs_thr [float]: calibrated conformity-score threshold
    '''
    conformity_scores, sorted_idxs = compute_conformity_scores(scores, l=l, kreg=kreg)
    box_id, true_class_ranking = np.where( sorted_idxs==np.expand_dims(gt_labels, axis=1) )
    cs_thr = np.quantile(conformity_scores[box_id, true_class_ranking], 1-alpha, method='higher')
    
    return cs_thr
    


def get_prediction_set(scores, threshold, l=0., kreg=0., gt_labels=None):
    '''
    Determine classes included in the prediction set for each image/bbox.

    Arguments:
    - scores [np.array]: predicted scores for every image/bbox (N_images, N_classes) in the validation/inference set
    - threshold [float]: calibrated conformity-score threshold.
    - l, kreg [float]: parameters used to regularise the conformity score, as introduced in 'https://arxiv.org/pdf/2009.14193.pdf'
    - gt_labels [np.array]: ground-truth class id of every images/bboxes (N_images). Provide if running validation, otherwise set to 'None'

    Output:
    - prediction_set_list [list(list(int))]: predicted class indices for each image/bbox
    - ranking_list [list(int)]: ranking of true classe's score, for each image/bbox (only returned if running validation)
    '''
    conformity_scores, sorted_idxs = compute_conformity_scores(scores, l=l, kreg=kreg)
    prediction_set_list = []
    ranking_list = []
    for cs, idxs in zip(conformity_scores, sorted_idxs): #loop over samples
        prediction_set = idxs[cs<=threshold]
        prediction_set_list.append(prediction_set)

        if gt_labels is not None:
            ranking_list.append(np.where(idxs==gt_label)[0].item())

    if gt_labels is not None:
        return prediction_set_list, ranking_list
    else:
        return prediction_set_list
