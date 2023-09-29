import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


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



def calibrate_cp_threshold(scores, gt_labels, alpha, l=0., kreg=0.):
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
    image_id, true_class_ranking = np.where( sorted_idxs==np.expand_dims(gt_labels, axis=1) )
    true_class_conformity_scores = conformity_scores[image_id, true_class_ranking] # conformity score of each image's true class

    # add random component
    U = np.random.random(true_class_conformity_scores.shape[0])
    true_class_conformity_scores -= U*scores[np.arange(scores.shape[0]), gt_labels]

    cs_thr = np.quantile(true_class_conformity_scores, 1-alpha, method='higher')
    
    return cs_thr, true_class_conformity_scores
    


def get_prediction_set(scores, threshold, calib_cs_distrib, l=0., kreg=0., gt_labels=None):
    '''
    Determine classes included in the prediction set for each image/bbox.

    Arguments:
    - scores [np.array]: predicted scores for every image/bbox (N_images, N_classes) in the validation/inference set
    - threshold [float]: calibrated conformity-score threshold
    - calib_cs_distrib [np.array]: true-classe's conformity score of every images in the calibration dataset
    - l, kreg [float]: parameters used to regularise the conformity score, as introduced in 'https://arxiv.org/pdf/2009.14193.pdf'
    - gt_labels [np.array]: ground-truth class id of every images/bboxes (N_images). Provide if running validation, otherwise set to 'None'

    Output:
    - prediction_set_list [list(np.array)]: predicted class indices for each image/bbox
    - size_list [list(int)]: size of prediction set, for each image/bbox
    - ranking_list [list(int)]: ranking of true classe's score, for each image/bbox (only returned if running validation)
    - covered_list [list(bool)]: whether true class in included in the prediction set, for each image/bbox (only returned if running validation)
    '''
    conformity_scores, sorted_idxs = compute_conformity_scores(scores, l=l, kreg=kreg)
    prediction_set_list = []
    ranking_list = []
    covered_list = []
    size_list = []
    credibility_list = []
    confidence_list = []
    confusion_matrix = np.zeros((scores.shape[1],scores.shape[1]))

    for cs, idxs, gt_label, score in zip(conformity_scores, sorted_idxs, gt_labels, scores): #loop over samples
        score = score[idxs] # sort scores in descending order
        
        n_selected = (cs<=threshold).sum() + 1
        # add random component
        U = np.random.random()
        reg = np.clip( l * (n_selected-kreg), 0, None )
        overshoot_ratio = (cs[n_selected-1] + reg - threshold) / (score[n_selected-1] + l * (n_selected > kreg) )
        # overshoot_ratio: how much conformity score the class that is just above threshold overshoots the threshold, relatively to the class prediction score
        if overshoot_ratio >= U:
            n_selected -= 1        
        
        prediction_set = idxs[:n_selected]
        if len(prediction_set)==0:
            prediction_set = np.array([idxs[0]])
        prediction_set_list.append(prediction_set)
            
        size_list.append(len(prediction_set))
        credibility_list.append( (calib_cs_distrib > cs[0]).sum() / len(calib_cs_distrib) )
        confidence_list.append( (calib_cs_distrib < cs[1]).sum() / len(calib_cs_distrib) )

        # fill confusion matrix --> which classes are associated together in prediction sets
        prediction_set_vector = np.zeros(scores.shape[1])
        prediction_set_vector[prediction_set] = 1
        confusion_matrix[prediction_set] += prediction_set_vector

        if gt_labels is not None:
            ranking_list.append(np.where(idxs==gt_label)[0].item())
            covered_list.append(gt_label.item() in prediction_set)
            
    if gt_labels is not None:
        return prediction_set_list, np.array(size_list), np.array(credibility_list), np.array(confidence_list), np.array(ranking_list), np.array(covered_list), confusion_matrix
    else:
        return prediction_set_list, np.array(size_list), np.array(credibility_list), np.array(confidence_list), confusion_matrix


    ######################################################################################################################################################################################
    ######################################################################## Plotting ####################################################################################################
    ######################################################################################################################################################################################


def plot_uncertainty_vs_difficulty(unc_type, ranking, uncertainty, target, classes, outpath):
    '''
    Compute and plot uncertainty metric (size, confidence, credibility) vs ranking of true class for each class + mean over all classes
    '''
    def get_uncertainty_vs_difficulty(unc_type, ranking, uncertainty, bins, max_ranking, outpath, plot_name_suffix='overall'):
        uncertainty_matrix = np.histogram2d(uncertainty, ranking, bins=[bins, np.arange(max_ranking+2)-0.5])[0]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlabel('true-class ranking')
        ax.set_ylabel(unc_type)

        ticks_unc = np.linspace(1, len(bins)-1, len(bins)-1).astype(int)
        c = ax.pcolor(uncertainty_matrix / uncertainty_matrix.sum(axis=0), cmap='Greens')
        for irow in range(uncertainty_matrix.shape[0]):
            for icol in range(uncertainty_matrix.shape[1]):
                ax.text(icol+0.5, irow+0.5, f'{int(uncertainty_matrix[irow][icol])}',
                           ha="center", va="center", color="black")

        mean = ticks_unc @ uncertainty_matrix / uncertainty_matrix.sum(axis=0)
        ticks_rank = np.linspace(0, max_ranking, max_ranking+1).astype(int)
        ax.plot(ticks_rank+0.5, mean, linestyle="None", marker='o', color='darkblue', label='mean', alpha=0.7)
        if unc_type=='credibility' or unc_type=='confidence':
            ax.plot(ticks_rank+0.5, [ np.median(uncertainty[ranking == idiff])*10 for idiff in range(max_ranking+1) ], linestyle="None", marker='o', color='red', label='median', alpha=0.7)
        else:
            ax.plot(ticks_rank+0.5, [ np.median(uncertainty[ranking == idiff]) for idiff in range(max_ranking+1) ], linestyle="None", marker='o', color='red', label='median', alpha=0.7)

        ax.set_xticks(ticks_rank+0.5)
        ax.set_xticklabels(ticks_rank, fontsize=10)
        if unc_type=='credibility' or unc_type=='confidence':
            ax.set_yticks(ticks_unc)
            ax.set_yticklabels(ticks_unc/10, fontsize=10)
        else:
            ax.set_yticks(ticks_unc-0.5)
            ax.set_yticklabels(ticks_unc, fontsize=10)

        ax.legend()
        cbar = fig.colorbar(c)
        cbar.set_label('p.d.f')
        fig.set_tight_layout(True)
        fig.savefig(f'{outpath}/{unc_type}_vs_ranking_{plot_name_suffix}.png')
        plt.close()
        
        return mean

    max_ranking = ranking.max()+1
    if unc_type=='size':
        bins = np.arange(uncertainty.max()+1)+0.5
    elif unc_type=='credibility' or unc_type=='confidence':
        bins = np.arange(11)/10
        
    mean_array_overall = get_uncertainty_vs_difficulty(unc_type, ranking, uncertainty, bins, max_ranking, outpath)

#    mean_array_list = []
#    for icls, cls in enumerate(classes):
#        mask = (target == icls)
#        mean_array = get_size_vs_difficulty(ranking[mask], size[mask], max_size, outpath, plot_name_suffix=cls)
#        mean_array_list.append(mean_array)
#
#    # plot average of classes of size vs ranking of true class
#    fig_svr, ax_svr = plt.subplots()
#    ax_svr.set_ylabel('prediction-set size')
#    ax_svr.set_xlabel('true-class ranking')
#
#    ax_svr.plot( range(max_size+1), mean_array_overall, linestyle="None", marker='p', color='black', label=f'overall', alpha=0.7)
#    ax_svr.plot( range(max_size+1), np.stack(mean_array_list).mean(axis=0), linestyle="None", marker='p', color='darkblue', label=f'average', alpha=0.7)
#
#    ax_svr.legend()
#    fig_svr.set_tight_layout(True)
#    fig_svr.savefig(f'{outpath}/average_size_vs_ranking.png')



def plot_coverage_per_class(covered, target, classes, alpha, outpath):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_ylabel('coverage')

    coverage_list = []
    coverage_list.append(covered.sum() / len(covered)) # overall coverage
    
    for icls, cls in enumerate(classes):
        mask = (target == icls)
        coverage = covered[mask].sum() / len(covered[mask])
        coverage_list.append(coverage)

    plt.bar(range(len(coverage_list)), coverage_list)
    ax.plot([-1, len(coverage_list)], [1-alpha, 1-alpha], color='red', linestyle='--', linewidth=1)

    ax.set_xticks( range(len(coverage_list)) )
    ax.set_xticklabels(['overall']+[c[:12] for c in classes], fontsize=10, rotation=45, ha='right')

    fig.set_tight_layout(True)
    fig.savefig(f'{outpath}/coverage_per_class.png')



def plot_coverage_vs_size(size, covered, target, classes, alpha, outpath):
    def get_coverage_vs_size(size, covered):
        size_list = []
        coverage_list = []
        n_sample = [] # number of samples per category
        for isize in range(size.max()):
            mask = (size == isize+1)
            coverage = covered[mask].sum() / len(covered[mask])
            n_sample.append(mask.sum())

            size_list.append(isize+1)
            coverage_list.append(coverage)

        return np.array(size_list), np.array(coverage_list), np.array(n_sample)
    
    # compute and plot coverage vs size
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('coverage')
    ax.set_xlabel('prediction-set size')

    # overall coverage
    size_list, coverage_list, n_sample = get_coverage_vs_size(size, covered)
    ax.scatter(size_list-0.45, coverage_list, 500*n_sample/n_sample.sum(), color='black', marker='o', label=f'Overall ({n_sample.sum()})')
    ax.plot([0.4, max(size_list)+0.5], [1-alpha, 1-alpha], color='red', linestyle='--', linewidth=1)

#    # coverage per class
#    markerstyle = ["p", "p", "p", "p", "p", "p", "p", "s", "p", "s", "s", "s", "*"]
#    for icls, cls in enumerate(classes):
#        mask = (target == icls)
#
#        size_list, coverage_list, n_sample = get_coverage_vs_size(size[mask], covered[mask])
#        ax.scatter(size_list-0.45+(icls+1)*0.064, coverage_list, 500*n_sample/n_sample.sum(), linestyle="None", marker=markerstyle[icls], label=f'{cls} ({n_sample.sum()})')
#        ax.plot([icls+1.5, icls+1.5], [-0.03, 1.03], color='black', linestyle='--', linewidth=1)

#    ax.legend(ncol=3, fontsize='small', framealpha=1)
    ax.set_xlim(0.5, max(size_list)+0.5)
    ax.set_ylim(-0.03, 1.03)
    fig.set_tight_layout(True)
    fig.savefig(f'{outpath}/coverage_vs_size.png')
    


def plot_confusion_matrix(confusion_matrix, n_sample, classes, outpath):
    '''
    Plot confusion matrix, which explains which classes are associated together in prediction sets.
    '''
    matrix_normalised = (confusion_matrix.T / np.diag(confusion_matrix)).T

    fig, ax = plt.subplots(1,2, sharey=True, gridspec_kw={'width_ratios':(0.7,12), 'wspace':0.05}, figsize=(16,11))
    opts = {'cmap': 'Greens', 'vmin': 0, 'vmax': +1}

    fraction_bboxes = (np.diag(confusion_matrix) / n_sample).reshape(len(classes),1)
    ax[0].pcolor( fraction_bboxes, **opts)
    for irow in range(fraction_bboxes.shape[0]):
        for icol in range(fraction_bboxes.shape[1]):
            ax[0].text(icol+0.5, irow+0.5, '{:.2f}'.format(fraction_bboxes[irow][icol]),
                       ha="center", va="center", color="black")
            ax[0].set_xticks([0.5])
            ax[0].set_xticklabels(['fraction of bboxes'], rotation=45, ha='right')

    heatmap = ax[1].pcolor(matrix_normalised, **opts)
    for irow in range(matrix_normalised.shape[0]):
        for icol in range(matrix_normalised.shape[1]):
            ax[1].text(icol+0.5, irow+0.5, '{:.2f}'.format(matrix_normalised[irow][icol]),
                       ha="center", va="center", color="black")

    ax[1].set_yticks(np.arange(0.5, matrix_normalised.shape[0], 1))
    ax[1].set_yticklabels(classes)
    ax[1].set_xticks(np.arange(0.5, matrix_normalised.shape[0], 1))
    ax[1].set_xticklabels(classes, rotation=45, ha='right')

    cbar = fig.colorbar(heatmap)
#    fig.set_tight_layout(True)
    fig.savefig(f'{outpath}/confusion_matrix.png')
