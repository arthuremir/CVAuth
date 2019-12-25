from cvauth.detectron2.detectron2.config import get_cfg as get_detectron_cfg

def setup_detectron_cfg(args):
    detectron_cfg = get_detectron_cfg()
    detectron_cfg.merge_from_file(args.config_file)
    detectron_cfg.merge_from_list(args.opts)
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    detectron_cfg.freeze()
    return detectron_cfg