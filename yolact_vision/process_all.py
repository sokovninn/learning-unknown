import os
import argparse

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Processing video using YOLACT')
    parser.add_argument('--objects_dir',
                        default='../Downloads/lu_dataset', type=str,
                        help='objects_dir')
    parser.add_argument('--proc',
                        default=1, type=int,
                        help='Process videos')
    parser.add_argument('--vis',
                        default=1, type=int,
                        help='Visualize results')
    parser.add_argument('--habitat_comparison', default=0, type=int,
                    help='Whether to draw comparion real with Habitat')

    args = parser.parse_args(argv)

    return args

#objects = ["backpack", "book", "computer", "dino", "extension", "keyboard", "laptop", "shoe", "skateboard", "ukulele"]

if __name__ == '__main__':
    args = parse_args()

    objects_filenames = os.listdir(args.objects_dir)

    for object_filename in objects_filenames:
        if args.proc:
            os.system("python proc_video.py --video=" + os.path.join(args.objects_dir, object_filename))
        if args.vis:
            os.system("python visualize_detection_results.py --habitat_comparison={} --plot_bars=0 --detections_dir=".format(args.habitat_comparison) + "video_frames_" + object_filename[:-4])
