import argparse


def make_parser():
    parser = argparse.ArgumentParser("OC-SORT parameters")

    # distributed
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=[800, 1440], nargs="+", type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.3,
        help="the iou threshold in Sort for matching",
    )
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
    parser.add_argument(
        "--inertia",
        type=float,
        default=0.2,
        help="the weight of VDC term in cost matrix",
    )
    parser.add_argument(
        "--deltat",
        type=int,
        default=3,
        help="time step difference to estimate direction",
    )
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--gt-type",
        type=str,
        default="_val_half",
        help="suffix to find the gt annotation",
    )
    parser.add_argument("--public", action="store_true", help="use public detection")
    parser.add_argument("--asso", default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")

    # for kitti/bdd100k inference with public detections
    parser.add_argument(
        "--raw_results_path",
        type=str,
        default="exps/permatrack_kitti_test/",
        help="path to the raw tracking results from other tracks",
    )
    parser.add_argument("--out_path", type=str, help="path to save output results")
    parser.add_argument(
        "--hp",
        action="store_true",
        help="use head padding to add the missing objects during \
            initializing the tracks (offline).",
    )

    # for demo video
    parser.add_argument("--demo_type", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("--path", default="./videos/demo.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser
