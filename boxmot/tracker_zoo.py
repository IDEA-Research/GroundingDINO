from pathlib import Path
from boxmot.strongsort.utils.parser import get_config


def get_tracker_config(tracker_type):
    tracking_config = \
        Path('./boxmot') /\
        tracker_type /\
        'configs' /\
        (tracker_type + '.yaml')
    return tracking_config
    

def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    
    cfg = get_config()
    cfg.merge_from_file(tracker_config)
    
    if tracker_type == 'strongsort':
        from boxmot.strongsort.strong_sort import StrongSORT
        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.strongsort.max_dist,
            max_iou_dist=cfg.strongsort.max_iou_dist,
            max_age=cfg.strongsort.max_age,
            max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
            n_init=cfg.strongsort.n_init,
            nn_budget=cfg.strongsort.nn_budget,
            mc_lambda=cfg.strongsort.mc_lambda,
            ema_alpha=cfg.strongsort.ema_alpha,

        )
        return strongsort
    
    elif tracker_type == 'ocsort':
        from boxmot.ocsort.ocsort import OCSort
        ocsort = OCSort(
            det_thresh=cfg.ocsort.det_thresh,
            max_age=cfg.ocsort.max_age,
            min_hits=cfg.ocsort.min_hits,
            iou_threshold=cfg.ocsort.iou_thresh,
            delta_t=cfg.ocsort.delta_t,
            asso_func=cfg.ocsort.asso_func,
            inertia=cfg.ocsort.inertia,
            use_byte=cfg.ocsort.use_byte,
        )
        return ocsort
    
    elif tracker_type == 'bytetrack':
        from boxmot.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=cfg.bytetrack.frame_rate
        )
        return bytetracker
    
    elif tracker_type == 'botsort':
        from boxmot.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            track_high_thresh=cfg.botsort.track_high_thresh,
            new_track_thresh=cfg.botsort.new_track_thresh,
            track_buffer =cfg.botsort.track_buffer,
            match_thresh=cfg.botsort.match_thresh,
            proximity_thresh=cfg.botsort.proximity_thresh,
            appearance_thresh=cfg.botsort.appearance_thresh,
            cmc_method =cfg.botsort.cmc_method,
            frame_rate=cfg.botsort.frame_rate,
            lambda_=cfg.botsort.lambda_
        )
        return botsort
    elif tracker_type == 'deepocsort':
        from boxmot.deepocsort.ocsort import OCSort
        deepocsort = OCSort(
            reid_weights,
            device,
            half,
            det_thresh=cfg.deepocsort.det_thresh,
            max_age=cfg.deepocsort.max_age,
            min_hits=cfg.deepocsort.min_hits,
            iou_threshold=cfg.deepocsort.iou_thresh,
            delta_t=cfg.deepocsort.delta_t,
            asso_func=cfg.deepocsort.asso_func,
            inertia=cfg.deepocsort.inertia,
        )
        return deepocsort
    else:
        print('No such tracker')
        exit()