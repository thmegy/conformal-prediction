import argparse
import glob, os, tqdm, json, sys
import numpy as np
from utils import inference_mmdet, compute_conformity_scores, calibrate_cp_threshold, get_prediction_set, blockPrint, enablePrint, plot_uncertainty_vs_difficulty, plot_coverage_per_class, plot_coverage_vs_size
from mmengine.config import Config
from mmengine.runner import Runner
import mmdet.apis



def predict(dataloader, detector):
    '''
    Predict bboxes and scores, and assign ground truth labels for all images in dataloader.
    '''
    predictions = []
    gt_bboxes_list = []
    gt_labels_list_full = [] # all gt-bboxes
    shape_list = []
    num_gts_list = []
    for i, images in enumerate(tqdm.tqdm(dataloader)):
        pred, gt_bboxes, gt_labels, shape, num_gts = inference_mmdet(images, detector)
        predictions += pred
        gt_bboxes_list += gt_bboxes
        gt_labels_list_full += gt_labels
        shape_list += shape
        num_gts_list += num_gts

    bboxes_list = []
    gt_labels_list = [] # only gt-bboxes matched to predicted bboxes
    gt_inds_list = []
    scores_list = []
    for pred in predictions:
        bboxes_list.append(pred.bboxes.cpu().detach().tolist())
        scores_list.append(pred.scores.cpu().detach().tolist())
        gt_labels_list.append(pred.gt_labels.cpu().detach().tolist())
        gt_inds_list.append(pred.gt_inds.cpu().detach().tolist())
        
    return bboxes_list, scores_list, gt_labels_list, gt_inds_list, gt_bboxes_list, gt_labels_list_full, shape_list, num_gts_list



def boxwise_fnr(scores_list, gt_inds_list, num_gts_list, score_thrs):
    '''
    Compute False Negative Rate as fraction of ground-truth object not detected, averaged over all images.
    '''
    fnr_array = []
    for scores, gt_inds, num_gts in zip(scores_list, gt_inds_list, num_gts_list):
        if num_gts > 0:
            gt_inds = np.array(gt_inds)
            scores = np.array(scores)
            if len(scores)==0:
                continue
            scores_sum = scores.sum(axis=1)

            fnr_list = []
            for thr in score_thrs:
                unique_gt_inds = np.unique(gt_inds[scores_sum > thr])
                unique_gt_inds = unique_gt_inds[unique_gt_inds>0] # remove index 0 --> abscence of gt

                fnr = 1 - len(unique_gt_inds) / num_gts # false negative rate
                fnr_list.append(fnr)

            fnr_array.append(fnr_list)

    fnr_array = np.array(fnr_array)
    return fnr_array



def pixelwise_fnr(bboxes_list, scores_list, gt_bboxes_list, shape_list, score_thrs):
    '''
    Compute False Negative Rate as fraction of ground-truth bboxes not covered by predicted bboxes.
    '''
    fnr_array = []
    for i, (pred_bboxes, scores, gt_bboxes, shape) in enumerate(tqdm.tqdm(zip(bboxes_list, scores_list, gt_bboxes_list, shape_list), total=len(bboxes_list))): # loop on images
        #if i > 100:
        #    break

        if len(gt_bboxes)==0:
            continue

        # make 2D map of pixels belonging to predicted bboxes (value of each pixel is the score of the bbox containing the pixel with the highest score)
        pred_array = np.zeros(shape)
        for pbox, s in zip(pred_bboxes, scores):
            x1, y1, x2, y2 = pbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            pred_array_tmp  = np.zeros(shape)
            pred_array_tmp[y1:y2, x1:x2] = sum(s)

            pred_array = np.where(pred_array_tmp > pred_array, pred_array_tmp, pred_array)

        # make 2D map of pixels belonging to ground-truth bboxes
        gt_array = np.zeros(shape, dtype=bool)
        for gtbox in gt_bboxes:
            x1, y1, x2, y2 = gtbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            gt_array[y1:y2, x1:x2] = True

        area = gt_array.sum()
        if area==0:
            continue

        fnr_list = []
        for thr in score_thrs:
            filtered_pred_array = pred_array > thr
            covered = (filtered_pred_array & gt_array).sum()

            fnr = 1 - covered / area
            fnr_list.append(fnr)

        fnr_array.append(fnr_list)

    fnr_array = np.array(fnr_array)
    return fnr_array



def multilabel_fnr(scores_list, gt_labels_list_full, score_thrs):
    fnr_array = []
    for scores, gt_labels in zip(scores_list, gt_labels_list_full):
        if len(gt_labels)==0:
            continue

        scores = np.array(scores)
        if len(scores)==0: # no predicted bboxes
            fnr_list = [1 for _ in score_thrs]
        else:
            max_scores = scores.max(axis=1)
            pred_labels = scores.argmax(axis=1)
            
            fnr_list = []
            for thr in score_thrs:
                unique_pred_labels = np.unique(pred_labels[max_scores>thr])
                covered_bool = np.isin(gt_labels, unique_pred_labels)
                fnr_list.append(1 - covered_bool.sum() / len(covered_bool))

            fnr_array.append(fnr_list)

    fnr_array = np.array(fnr_array)
    return fnr_array
    



def main(args):
    config = Config.fromfile(args.config)
    config.work_dir = 'outputs/conformal_prediction/'
    config.load_from = args.checkpoint
    
    blockPrint()
    runner = Runner.from_cfg(config)
    enablePrint()


    detector = mmdet.apis.init_detector(
        args.config,
        args.checkpoint,
        device=f'cuda:{args.gpu_id}',
    )

    # calibration
    calib_loader = runner.val_dataloader

    if args.skip_inference:
        with open(f'{args.outpath}/results_inference_calib.json', 'r') as fin:
            res_dict = json.load(fin)
            bboxes_list = res_dict['bboxes']
            scores_list = res_dict['scores']
            gt_labels_list = res_dict['gt_labels']
            gt_inds_list = res_dict['gt_inds']
            gt_bboxes_list = res_dict['gt_bboxes']
            gt_labels_list_full = res_dict['gt_labels_full']
            shape_list = res_dict['shape']
            num_gts_list = res_dict['num_gt']
    else:
        bboxes_list, scores_list, gt_labels_list, gt_inds_list, gt_bboxes_list, gt_labels_list_full, shape_list, num_gts_list = predict(calib_loader, detector)
        with open(f'{args.outpath}/results_inference_calib.json', 'w') as fout:
            results = {
                'bboxes' : bboxes_list,
                'scores' : scores_list,
                'gt_labels' : gt_labels_list,
                'gt_inds' : gt_inds_list,
                'gt_bboxes' : gt_bboxes_list,
                'gt_labels_full' : gt_labels_list_full,
                'shape' : shape_list,
                'num_gt' : num_gts_list
            }
            json.dump(results, fout, indent = 6)
            
    score_thrs = np.linspace(0,1,21)[1:-1]


    if args.skip_fnr_computation:
        fnr_array = np.load(f'{args.outpath}/fnr_calib_{args.fnr_type}wise.npy')
    else:
        if args.fnr_type=='pixel':
            fnr_array = pixelwise_fnr(bboxes_list, scores_list, gt_bboxes_list, shape_list, score_thrs)
        elif args.fnr_type=='box':
            fnr_array = boxwise_fnr(scores_list, gt_inds_list, num_gts_list, score_thrs)
        elif args.fnr_type=='multilabel':
            fnr_array = multilabel_fnr(scores_list, gt_labels_list_full, score_thrs)
            
        fnr_array = np.array(fnr_array)
        with open(f'{args.outpath}/fnr_calib_{args.fnr_type}wise.npy', 'wb') as fout:
            np.save(fout, fnr_array)

    print(fnr_array)
    # conformal risk control
    n = fnr_array.shape[0]
    b = 1
    modified_fnr = (fnr_array.mean(axis=0)*n + b) / (n+1) # formula from Conformal Risk Control paper
    score_thr_args = np.where( modified_fnr <= args.alpha )[0]
    if len(score_thr_args)==0:
        print(f'Minimum modified FNR = {modified_fnr.min():.3f}. It is larger than the chosen significance level alpha = {args.alpha}.')
        sys.exit('Exiting')
    score_thr_arg = score_thr_args[-1]
    score_thr = score_thrs[score_thr_arg]
    print(score_thr_arg, score_thr)


    # validation
    if args.skip_inference:
        with open(f'{args.outpath}/results_inference_test.json', 'r') as fin:
            res_dict = json.load(fin)
            bboxes_list = res_dict['bboxes']
            scores_list = res_dict['scores']
            gt_labels_list = res_dict['gt_labels']
            gt_inds_list = res_dict['gt_inds']
            gt_bboxes_list = res_dict['gt_bboxes']
            gt_labels_list_full = res_dict['gt_labels_full']
            shape_list = res_dict['shape']
            num_gts_list = res_dict['num_gt']
    else:
        test_loader = runner.test_dataloader
        bboxes_list, scores_list, gt_labels_list, gt_inds_list, gt_bboxes_list, gt_labels_list_full, shape_list, num_gts_list = predict(test_loader, detector)
        with open(f'{args.outpath}/results_inference_test.json', 'w') as fout:
            results = {
                'bboxes' : bboxes_list,
                'scores' : scores_list,
                'gt_labels' : gt_labels_list,
                'gt_inds' : gt_inds_list,
                'gt_bboxes' : gt_bboxes_list,
                'gt_labels_full' : gt_labels_list_full,
                'shape' : shape_list,
                'num_gt' : num_gts_list
            }
            json.dump(results, fout, indent = 6)

    if args.fnr_type=='pixel':
        fnr_array = pixelwise_fnr(bboxes_list, scores_list, gt_bboxes_list, shape_list, [score_thr])
    elif args.fnr_type=='box':
        fnr_array = boxwise_fnr(scores_list, gt_inds_list, num_gts_list, [score_thr])
    elif args.fnr_type=='multilabel':
        fnr_array = multilabel_fnr(scores_list, gt_labels_list_full, [score_thr])

    fnr_array = np.array(fnr_array)

    print(fnr_array.mean(axis=0))

#    scores, gt_labels = predict(test_loader, inferencer)
#    prediction_set_list, size, credibility, confidence, ranking, covered = get_prediction_set(scores, cs_thr, true_class_conformity_scores, l=args.l, kreg=args.kreg, gt_labels=gt_labels)
#
#    np.save(f'{args.outpath}/true_class_conformity_scores_calib.npy', true_class_conformity_scores)
#
#    argmax = scores.argmax(axis=1)
#    print(compute_accuracy(argmax,gt_labels))
#    print(covered.sum() / len(covered))
#    print(np.unique(size, return_counts=True))
#
#    plot_uncertainty_vs_difficulty('size', ranking, size, gt_labels, classes, args.outpath)
#    plot_uncertainty_vs_difficulty('credibility', ranking, credibility, gt_labels, classes, args.outpath)
#    plot_uncertainty_vs_difficulty('confidence', ranking, confidence, gt_labels, classes, args.outpath)
#    plot_coverage_per_class(covered, gt_labels, classes, args.alpha, args.outpath)
#    plot_coverage_vs_size(size, covered, gt_labels, classes, args.alpha, args.outpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmpretrain config")
    parser.add_argument("--checkpoint", required=True, help="mmpretrain checkpoint")
    parser.add_argument("--im-dir", help="Directory containing the images")
    parser.add_argument("--outpath", required=True, help="Path to output directory")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    
    parser.add_argument("--alpha", type=float, default=0.1, help="significance level")
    parser.add_argument("--l", type=float, default=0., help="lambda parameter for regularisation of conformality score")
    parser.add_argument("--kreg", type=float, default=0., help="kreg parameter for regularisation of conformality score")

    parser.add_argument("--fnr-type", choices=['pixel', 'box', 'multilabel'], required=True, help="pixelwise or boxwise False Negative Rate")

    parser.add_argument("--skip-inference", action='store_true', help="Skip inference. Load results previously saved.")
    parser.add_argument("--skip-fnr-computation", action='store_true', help="Skip determination of False Negative Rate. Load results previously saved.")

#    parser.add_argument("--per-class-thr", action='store_true', help="Determine and apply per-class conformity-score thresholds.")
    
#    parser.add_argument('--post-process', action='store_true', help='Make summary plots without processing videos again.')
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    
    main(args)
