import argparse
import glob, os, tqdm
import numpy as np
from utils import inference_mmpretrain, compute_conformity_scores, calibrate_cp_threshold, get_prediction_set, blockPrint, enablePrint
from mmengine.config import Config
from mmengine.runner import Runner
from mmpretrain import ImageClassificationInferencer


def predict(dataloader, inferencer):
    '''
    Predict scores and retrieve ground truth labels for all images in dataloader.
    '''
    scores = []
    gt_labels = []
    for img_batch in tqdm.tqdm(dataloader):
        metadata = img_batch['data_samples']
        images = [m.img_path for m in metadata]
        pred = inference_mmpretrain(images, inferencer)
        scores.append(pred)
        gt_labels += [m.gt_label.item() for m in metadata]
        

    scores = np.concatenate(scores, axis=0)
    gt_labels = np.array(gt_labels)

    return scores, gt_labels



def compute_accuracy(argmax, target):
    return (argmax==target).sum() / len(target)



def main(args):
    config = Config.fromfile(args.config)
    config.work_dir = 'outputs/conformal_prediction/'
    config.load_from = args.checkpoint
    
    blockPrint()
    runner = Runner.from_cfg(config)
    enablePrint()


    inferencer = ImageClassificationInferencer(
        model=args.config,
        pretrained=args.checkpoint,
        device=f'cuda:{args.gpu_id}'
    )

    # calibration
    calib_loader = runner.val_dataloader
    scores, gt_labels = predict(calib_loader, inferencer)    
    cs_thr = calibrate_cp_threshold(scores, gt_labels, args.alpha, l=args.l, kreg=args.kreg)
    print(cs_thr)


    # validation
    test_loader = runner.test_dataloader
    scores, gt_labels = predict(test_loader, inferencer)
    prediction_set_list, ranking, covered, size = get_prediction_set(scores, cs_thr, l=args.l, kreg=args.kreg, gt_labels=gt_labels)

    argmax = scores.argmax(axis=1)
    print(compute_accuracy(argmax,gt_labels))
    print(covered.sum() / len(covered))
    print(np.unique(size, return_counts=True))



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
#    parser.add_argument("--per-class-thr", action='store_true', help="Determine and apply per-class conformity-score thresholds.")
    
#    parser.add_argument("--skip-calibration", action='store_true', help="Skip calibration and run directly validation.")
#    parser.add_argument('--post-process', action='store_true', help='Make summary plots without processing videos again.')
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    
    main(args)
