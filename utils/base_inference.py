def inference_mmpretrain(images, config, checkpoint, device='cuda:0'):
    '''
    Run inference on images with a trained classification model from mmpretrain.

    Arguments:
    - image [list(str)]: list of paths to images to run inference on
    - config [str]: path to config file of model
    - checkpoint [str]: path to trained weights of model
    - device [str]: device to run inference on
    '''
    from mmpretrain import ImageClassificationInferencer

    inferencer = ImageClassificationInferencer(
        model=config,
        pretrained=checkpoint,
        device=device
    )
    
    results = inferencer(images)

    scores = 

    return scores
