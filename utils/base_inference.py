import numpy as np
from .misc import blockPrint, enablePrint



def inference_mmpretrain(images, inferencer):
    '''
    Run inference on images with a trained classification model from mmpretrain.

    Arguments:
    - images [list(str)]: list of paths to images to run inference on
    - inferencer [mmpretrain.ImageClassificationInferencer]: loaded classification model 

    Outputs:
    - scores [np.array]: predicted scores for all classes for each image (N_images, N_classes)
    '''
    
    blockPrint()
    results = inferencer(images)
    enablePrint()

    scores = [res['pred_scores'] for res in results]
    scores = np.array(scores)

    return scores

