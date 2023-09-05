import numpy as np
from .misc import blockPrint, enablePrint
import torch, copy
import mmdet.apis
import mmdet.utils
from mmdet.structures.bbox import cat_boxes, get_box_tensor, get_box_wh, scale_boxes
from mmcv.transforms import Compose
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmengine.config import Config
from mmengine.runner import Runner



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



def inference_mmdet(images, detector):
    '''
    Predict the bboxes and scores for a batch of images.
    Returns bboxes, scores, predicted labels and ground-truth labels.
    '''
    img_batch = detector.data_preprocessor(images, False)
    with torch.no_grad():
        results = detector(**img_batch, mode='tensor')

    # from bbox_head.predict function
    batch_img_metas = [
        data_samples.metainfo for data_samples in img_batch['data_samples']
    ]

    # extract GT bboxes
    targets = mmdet.models.utils.unpack_gt_instances(img_batch['data_samples'])

    gt_bboxes_list = []
    for im in targets[0]:
        gt_bboxes_list.append(im.bboxes.cpu().detach().tolist())
        
    
    # get bboxes and corresponding scores and ground truth, filtered by score and nms
    predictions, num_gts = predict_by_feat(detector, targets, *results, batch_img_metas=batch_img_metas, rescale=True)

#    print_bboxes(targets[0][0]['bboxes'].cpu().detach().numpy(),
#                 predictions[0]['bboxes'].cpu().detach().numpy(),
#                 predictions[0]['gt_labels'].cpu().detach().numpy(),
#                 targets[2][0]['img_path']
#                 )

    return predictions, gt_bboxes_list, num_gts



def print_bboxes(target_bboxes, pred_bboxes, pred_labels, name):
    import cv2
    image = cv2.imread(name)
    name = name.split('/')[-1]
    
    for tbox in target_bboxes:
        cv2.rectangle(image, tbox[:2].astype(int), tbox[2:].astype(int), (0,0,255), 5)

    for pbox, plab in zip(pred_bboxes, pred_labels):
        if plab == -1:
            cv2.rectangle(image, pbox[:2].astype(int), pbox[2:].astype(int), (255,0,0), 5)
        else:
            cv2.rectangle(image, pbox[:2].astype(int), pbox[2:].astype(int), (0,255,0), 5)

    cv2.imwrite(f'test/{name}', image)


    

def predict_by_feat(model, targets, cls_scores, bbox_preds, score_factors = None,
                    batch_img_metas = None, cfg = None,
                    rescale = False, with_nms = True):
    """Transform a batch of output features extracted from the head into
    bbox results.

    Note: When score_factors is not None, the cls_scores are
    usually multiplied by it then obtain the real score used in NMS,
    such as CenterNess in FCOS, IoU branch in ATSS.

    Args:
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        score_factors (list[Tensor], optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W). Defaults to None.
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        list[:obj:`InstanceData`]: Object detection results of each image
        after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    assert len(cls_scores) == len(bbox_preds)

    if score_factors is None:
        # e.g. Retina, FreeAnchor, Foveabox, etc.
        with_score_factors = False
    else:
        # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
        with_score_factors = True
        assert len(cls_scores) == len(score_factors)

    num_levels = len(cls_scores)

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = model.bbox_head.prior_generator.grid_priors(
        featmap_sizes,
        dtype=cls_scores[0].dtype,
        device=cls_scores[0].device)

    # get targets
    (batch_gt_instances, batch_gt_instances_ignore,
     batch_img_metas) = targets

    result_list = []
    num_gts_list = []

    for img_id in range(len(batch_img_metas)):
        img_meta = batch_img_metas[img_id]
        cls_score_list = mmdet.models.utils.select_single_mlvl(
            cls_scores, img_id, detach=True)
        bbox_pred_list = mmdet.models.utils.select_single_mlvl(
            bbox_preds, img_id, detach=True)
        if with_score_factors:
            score_factor_list = mmdet.models.utils.select_single_mlvl(
                score_factors, img_id, detach=True)
        else:
            score_factor_list = [None for _ in range(num_levels)]

        results, num_gts = _predict_by_feat_single(
            model,
            gt_instances=batch_gt_instances[img_id],
            gt_instances_ignore=batch_gt_instances_ignore[img_id],            
            cls_score_list=cls_score_list,
            bbox_pred_list=bbox_pred_list,
            score_factor_list=score_factor_list,
            mlvl_priors=mlvl_priors,
            img_meta=img_meta,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms)
        result_list.append(results)
        num_gts_list.append(num_gts)
    return result_list, num_gts_list



def _predict_by_feat_single(model, gt_instances, gt_instances_ignore, cls_score_list, bbox_pred_list, score_factor_list,
                            mlvl_priors, img_meta, cfg, rescale = False,
                            with_nms = True):
    """Transform a single image's features extracted from the head into
    bbox results.

    Args:
        cls_score_list (list[Tensor]): Box scores from all scale
            levels of a single image, each item has shape
            (num_priors * num_classes, H, W).
        bbox_pred_list (list[Tensor]): Box energies / deltas from
            all scale levels of a single image, each item has shape
            (num_priors * 4, H, W).
        score_factor_list (list[Tensor]): Score factor from all scale
            levels of a single image, each item has shape
            (num_priors * 1, H, W).
        mlvl_priors (list[Tensor]): Each element in the list is
            the priors of a single level in feature pyramid. In all
            anchor-based methods, it has shape (num_priors, 4). In
            all anchor-free methods, it has shape (num_priors, 2)
            when `with_stride=True`, otherwise it still has shape
            (num_priors, 4).
        img_meta (dict): Image meta info.
        cfg (mmengine.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        :obj:`InstanceData`: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    if score_factor_list[0] is None:
        # e.g. Retina, FreeAnchor, etc.
        with_score_factors = False
    else:
        # e.g. FCOS, PAA, ATSS, etc.
        with_score_factors = True

    cfg = model.bbox_head.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)
    img_shape = img_meta['img_shape']

    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_scores = []
    mlvl_logits = []
    if with_score_factors:
        mlvl_score_factors = []
    else:
        mlvl_score_factors = None
    for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
            enumerate(zip(cls_score_list, bbox_pred_list,
                          score_factor_list, mlvl_priors)):

        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        dim = model.bbox_head.bbox_coder.encode_size
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
        if with_score_factors:
            score_factor = score_factor.permute(1, 2,
                                                0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2,
                                      0).reshape(-1, model.bbox_head.cls_out_channels)
        
        if model.bbox_head.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            # remind that we set FG labels to [0, num_class-1]
            # since mmdet v2.0
            # BG cat_id: num_class
            scores = cls_score.softmax(-1)[:, :-1]

        mlvl_bbox_preds.append(bbox_pred)
        mlvl_valid_priors.append(priors)
        mlvl_scores.append(scores)
        mlvl_logits.append(cls_score)

        if with_score_factors:
            mlvl_score_factors.append(score_factor)

    bbox_pred = torch.cat(mlvl_bbox_preds)
    priors = cat_boxes(mlvl_valid_priors)
    bboxes = model.bbox_head.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
    if rescale:
        assert img_meta.get('scale_factor') is not None
        scale_factor = [1 / s for s in img_meta['scale_factor']]
        bboxes = scale_boxes(bboxes, scale_factor)

    num_level_priors = [len(s) for s in mlvl_scores]
    mlvl_scores = torch.cat(mlvl_scores)
    mlvl_score_factors = torch.cat(mlvl_score_factors)
    mlvl_logits = torch.cat(mlvl_logits)

    # apply scale factors, e.g. centerness for anchor-free detectors
    if with_score_factors:
        mlvl_scores = (mlvl_scores.T * mlvl_score_factors).T
    
    # assign a ground truth label to each predicted bbox
    assign_result = model.bbox_head.assigner.assign(InstanceData(priors=bboxes), num_level_priors,
                                                    gt_instances, gt_instances_ignore)
    gt_labels = assign_result.labels
    gt_inds = assign_result.gt_inds


    # filter prediction by score, and topk
    max_scores, labels = torch.max(mlvl_scores, 1)
    score_thr = cfg.get('score_thr', 0)
    valid_mask = max_scores > score_thr
    valid_idxs = torch.nonzero(valid_mask) 
    
    nms_pre = cfg.get('nms_pre', -1)
    num_topk = min(nms_pre, valid_idxs.size(0))
    sorted_scores, idxs = max_scores[valid_mask].sort(descending=True)
    topk_idxs = valid_idxs[idxs[:num_topk]].squeeze()

    if topk_idxs.size() == torch.Size([]): # case only 1 idx left, tensor is just a number
        topk_idxs = topk_idxs.unsqueeze(0)

    mlvl_scores = mlvl_scores[topk_idxs]
    mlvl_logits = mlvl_logits[topk_idxs]
    mlvl_labels = labels[topk_idxs]
    gt_labels = gt_labels[topk_idxs]
    gt_inds = gt_inds[topk_idxs]
    bboxes = bboxes[topk_idxs]
        
    results = InstanceData()
    results.bboxes = bboxes
    results.scores = mlvl_scores
    results.logits = mlvl_logits
    results.labels = mlvl_labels
    results.gt_labels = gt_labels
    results.gt_inds = gt_inds


    # filter small size bboxes
    if cfg.get('min_bbox_size', -1) >= 0:
        w, h = get_box_wh(results.bboxes)
        valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
        if not valid_mask.all():
            results = results[valid_mask]

    # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
    if with_nms and results.bboxes.numel() > 0:
        bboxes = get_box_tensor(results.bboxes)
        det_bboxes, keep_idxs = batched_nms(bboxes, results.scores.max(dim=1)[0],
                                            results.labels, cfg.nms)
        results = results[keep_idxs]
        # some nms would reweight the score, such as softnms
        #results.scores = det_bboxes[:, -1]
        results = results[:cfg.max_per_img]

    return results, assign_result.num_gts
