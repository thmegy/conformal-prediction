import argparse
import glob, os, tqdm, json, sys
import numpy as np
from utils import inference_mmdet, compute_conformity_scores, calibrate_cp_threshold, get_prediction_set, blockPrint, enablePrint
from utils import plot_uncertainty_vs_difficulty, plot_coverage_per_class, plot_coverage_vs_size, plot_confusion_matrix
from mmengine.config import Config
from mmengine.runner import Runner
import mmdet.apis
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')



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
    gt_labels_list_matched = [] # only gt-bboxes matched to predicted bboxes
    gt_inds_list = []
    scores_list = []
    for pred in predictions:
        bboxes_list.append(pred.bboxes.cpu().detach().tolist())
        scores_list.append(pred.scores.cpu().detach().tolist())
        gt_labels_list_matched.append(pred.gt_labels.cpu().detach().tolist())
        gt_inds_list.append(pred.gt_inds.cpu().detach().tolist())
        
    return bboxes_list, scores_list, gt_labels_list_matched, gt_inds_list, gt_bboxes_list, gt_labels_list_full, shape_list, num_gts_list



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



def get_inference(args, loader, dataset):
    if args.skip_inference:
        with open(f'{args.outpath}/results_inference_{dataset}.json', 'r') as fin:
            results = json.load(fin)
    else:
        bboxes_list, scores_list, gt_labels_list_matched, gt_inds_list, gt_bboxes_list, gt_labels_list_full, shape_list, num_gts_list = predict(loader, detector)
        with open(f'{args.outpath}/results_inference_{dataset}.json', 'w') as fout:
            results = {
                'bboxes' : bboxes_list,
                'scores' : scores_list,
                'gt_labels' : gt_labels_list_matched,
                'gt_inds' : gt_inds_list,
                'gt_bboxes' : gt_bboxes_list,
                'gt_labels_full' : gt_labels_list_full,
                'shape' : shape_list,
                'num_gt' : num_gts_list
            }
            json.dump(results, fout, indent = 6)

    return results



def get_fnr(args, results, score_thrs, dataset):
    
    bboxes_list = results['bboxes']
    scores_list = results['scores']
    gt_inds_list = results['gt_inds']
    gt_bboxes_list = results['gt_bboxes']
    gt_labels_list_full = results['gt_labels_full']
    shape_list = results['shape']
    num_gts_list = results['num_gt']
            
    if args.skip_fnr_computation:
        fnr_array = np.load(f'{args.outpath}/{args.fnr_type}wise-fnr/fnr_{dataset}.npy')
    else:
        if args.fnr_type=='pixel':
            fnr_array = pixelwise_fnr(bboxes_list, scores_list, gt_bboxes_list, shape_list, score_thrs)
        elif args.fnr_type=='box':
            fnr_array = boxwise_fnr(scores_list, gt_inds_list, num_gts_list, score_thrs)
        elif args.fnr_type=='multilabel':
            fnr_array = multilabel_fnr(scores_list, gt_labels_list_full, score_thrs)
            
        fnr_array = np.array(fnr_array)
        with open(f'{args.outpath}/{args.fnr_type}wise-fnr/fnr_{dataset}.npy', 'wb') as fout:
            np.save(fout, fnr_array)

    return fnr_array



def filter_results_for_cp(results, score_thr_crc, dataset):
    scores_list = results['scores']
    gt_labels_list_matched = results['gt_labels']

    # modify structure from (image, bboxes) to (bboxes) --> flatten
    scores = []
    gt_labels = []
    for s, l in zip(scores_list, gt_labels_list_matched):
        scores += s
        gt_labels += l
    
    scores = np.array(scores)
    gt_labels = np.array(gt_labels)

    # filter bboxes based on CRC score-threshold
    mask_score_sum = (scores.sum(axis=1) > score_thr_crc)
    scores_filtered = scores[mask_score_sum]
    gt_labels_filtered = gt_labels[mask_score_sum]

    mask_matched = gt_labels_filtered != -1 # select pred bboxes matched to a gt bbox
    print(f'{mask_score_sum.sum()} remaining predicted bboxes in {dataset} dataset')
    print(f'{mask_matched.sum()} matched, {mask_score_sum.sum()-mask_matched.sum()} unmatched')
        
    # apply inverse sigmoid, then softmax to scores
    scores_filtered = np.log(scores_filtered / (1-scores_filtered))
    scores_filtered = (np.exp(scores_filtered).T / np.exp(scores_filtered).sum(axis=1)).T

    return scores_filtered, gt_labels_filtered, mask_matched


    



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

    #############################################################################
    ####################### conformal risk control ##############################
    #############################################################################

    ## calibration ##
    blockPrint()
    calib_loader = runner.val_dataloader
    enablePrint()
    results_calib = get_inference(args, calib_loader, 'calib')
            
    score_thrs = np.linspace(0,1,81)[1:-1]
    fnr_array = get_fnr(args, results_calib, score_thrs, 'calib')

    n = fnr_array.shape[0]
    b = 1
    modified_fnr = (fnr_array.mean(axis=0)*n + b) / (n+1) # formula from Conformal Risk Control paper
    score_thr_args = np.where( modified_fnr <= args.alpha_crc )[0]
    
    if len(score_thr_args)==0:
        print(f'Minimum modified FNR = {modified_fnr.min():.3f}. It is larger than the chosen significance level alpha = {args.alpha_crc}.')
        sys.exit('Exiting')
    score_thr_arg = score_thr_args[-1]
    score_thr = score_thrs[score_thr_arg]
    print('')
    print('CRC Calibration:')
    print(f'minimum FNR = {modified_fnr.min():.3f}')
    print(f'score threshold = {score_thr}')
    print(f'FNR for alpha={args.alpha_crc} = {fnr_array.mean(axis=0)[score_thr_arg]:.3f}')
    print('')

    # plot lambda calibration curve
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'FNR')
    ax.plot(score_thrs, fnr_array.mean(axis=0))

    fig.set_tight_layout(True)
    fig.savefig(f'{args.outpath}/{args.fnr_type}wise-fnr/FNR_vs_score_thr_calib.png')

    ## validation ##
    blockPrint()
    test_loader = runner.test_dataloader
    enablePrint()
    results_test = get_inference(args, test_loader, 'test')

    fnr_array = get_fnr(args, results_test, [score_thr], 'test')

    print('')
    print('CRC Test:')
    print(f'FNR for alpha={args.alpha_crc} = {fnr_array.mean(axis=0)[0]:.3f}')
    print('')


    #################################################################################################################################
    ###################################################### Apply CP to remaining bboxes #############################################
    #################################################################################################################################

    ## calibration ##
    scores_filtered_calib, gt_labels_filtered_calib, mask_matched_calib = filter_results_for_cp(results_calib, score_thr, 'calib')

    cs_thr, true_class_conformity_scores = calibrate_cp_threshold(scores_filtered_calib[mask_matched_calib], gt_labels_filtered_calib[mask_matched_calib], args.alpha_cp, l=args.l, kreg=args.kreg)
    np.save(f'{args.outpath}/{args.fnr_type}wise-fnr/true_class_conformity_scores_calib.npy', true_class_conformity_scores)

    print('')
    print(f'Conformity score threshold = {cs_thr:.3f}')
    print('')

    
    ## validation ##
    scores_filtered_test, gt_labels_filtered_test, mask_matched_test = filter_results_for_cp(results_test, score_thr, 'test')
    
    prediction_set_list, size, credibility, confidence, ranking, covered, confusion_matrix = get_prediction_set(scores_filtered_test, cs_thr,
                                                                                                                true_class_conformity_scores, l=args.l,
                                                                                                                kreg=args.kreg, gt_labels=gt_labels_filtered_test)

    print(f'average set-size for matched bboxes = {size[mask_matched_test].mean():.3f}')
    print(f'average set-size for unmatched bboxes = {size[~mask_matched_test].mean():.3f}')
    print('')
    
    classes = test_loader.dataset.metainfo['classes']
    plot_uncertainty_vs_difficulty('size',
                                   ranking,
                                   size[mask_matched_test],
                                   gt_labels_filtered_test,
                                   classes,
                                   f'{args.outpath}/{args.fnr_type}wise-fnr')
    plot_coverage_per_class(covered,
                            gt_labels_filtered_test[mask_matched_test],
                            classes,
                            args.alpha_cp,
                            f'{args.outpath}/{args.fnr_type}wise-fnr')
    plot_coverage_vs_size(size[mask_matched_test],
                          covered,
                          gt_labels_filtered_test[mask_matched_test],
                          classes,
                          args.alpha_cp,
                          f'{args.outpath}/{args.fnr_type}wise-fnr')
    plot_confusion_matrix(confusion_matrix,
                          len(size),
                          classes,
                          f'{args.outpath}/{args.fnr_type}wise-fnr')


    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel('size')
    bins = np.arange(size.max())+1
    ax.hist(size[mask_matched_test], bins=bins, density=True, alpha=0.5, label='matched bboxes')
    ax.hist(size[~mask_matched_test], bins=bins, density=True, alpha=0.5, label='unmatched bboxes')

    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(f'{args.outpath}/{args.fnr_type}wise-fnr/size_distribution.png')

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmpretrain config")
    parser.add_argument("--checkpoint", required=True, help="mmpretrain checkpoint")
    parser.add_argument("--im-dir", help="Directory containing the images")
    parser.add_argument("--outpath", required=True, help="Path to output directory")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    
    parser.add_argument("--alpha_crc", type=float, default=0.25, help="significance level for Conformal Risk Control")
    parser.add_argument("--alpha_cp", type=float, default=0.1, help="significance level for Conformal Prediction")
    parser.add_argument("--l", type=float, default=0., help="lambda parameter for regularisation of conformality score")
    parser.add_argument("--kreg", type=float, default=0., help="kreg parameter for regularisation of conformality score")

    parser.add_argument("--fnr-type", choices=['pixel', 'box', 'multilabel'], required=True, help="pixelwise or boxwise False Negative Rate")

    parser.add_argument("--skip-inference", action='store_true', help="Skip inference. Load results previously saved.")
    parser.add_argument("--skip-fnr-computation", action='store_true', help="Skip determination of False Negative Rate. Load results previously saved.")

#    parser.add_argument("--per-class-thr", action='store_true', help="Determine and apply per-class conformity-score thresholds.")
    
#    parser.add_argument('--post-process', action='store_true', help='Make summary plots without processing videos again.')
    args = parser.parse_args()

    os.makedirs(f'{args.outpath}/{args.fnr_type}wise-fnr', exist_ok=True)
    
    main(args)
