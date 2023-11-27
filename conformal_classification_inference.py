import argparse
import glob, os, tqdm, json
import numpy as np
from utils import inference_mmpretrain, compute_conformity_scores, get_prediction_set, blockPrint, enablePrint
from mmengine.config import Config
from mmengine.runner import Runner
from mmpretrain import ImageClassificationInferencer



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

    # get list of images in directory
    images = glob.glob(f'{args.im_dir}/*jpg')

    # get calibration results
    true_class_conformity_scores = np.load(f'{args.calibpath}/true_class_conformity_scores_calib.npy')
    with open(f'{args.calibpath}/calibration_results.json', 'r') as fin:
        calibration_results = json.load(fin)
        cs_thr = calibration_results['conformality_score_thr']
        lambda_reg = calibration_results['regularisation']['lambda']
        kreg = calibration_results['regularisation']['kreg']


    #classes = test_loader.dataset.CLASSES
    
    # run inference
    scores = []
    for img in tqdm.tqdm(images):
        pred = inference_mmpretrain(img, inferencer)
        scores.append(pred)        
    scores = np.concatenate(scores, axis=0)

    # get prediction sets
    prediction_set_list, size, credibility, confidence, confusion_matrix = get_prediction_set(scores, cs_thr, true_class_conformity_scores, l=lambda_reg, kreg=kreg, gt_labels=None)

    print(size)
    results = {
        'prediction_set_list' : prediction_set_list
    }
    
    with open(f'{args.outpath}/results.json', 'w') as fout:
        json.dump(results, fout, indent = 6)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="mmpretrain config")
    parser.add_argument("--checkpoint", required=True, help="mmpretrain checkpoint")
    parser.add_argument("--im-dir", help="Directory containing the images")
    parser.add_argument("--calibpath", required=True, help="Path to calibration results")
    parser.add_argument("--outpath", required=True, help="Path to output directory")
    parser.add_argument("--gpu-id", default='0', help="ID of gpu to be used")
    
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    
    main(args)
