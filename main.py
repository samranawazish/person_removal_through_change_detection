# Name: Samra Nawazish
# Assignment Number: 01


import argparse
from changeDetection import changeDetection


def main():
    
    parser = argparse.ArgumentParser(description="Motion Detection using Classical CV Methods")
    parser.add_argument(
        '--input_folder', type=str, required=True,
        help='Path to input folder containing frames (test images)'
    )
    parser.add_argument(
        '--output_folder', type=str, required=True,
        help='Path to save results (binary masks and final video)'
    )
    parser.add_argument(
        '--input_ext', type=str, default='png',
        help='Extension of input images (png, jpg, jpeg)'
    )
    parser.add_argument(
        '--output_ext', type=str, default='png',
        help='Extension of output masks (png, jpg)'
    )
    parser.add_argument(
        '--video_format', type=str, default='mp4',
        help='Video format for saving output (mp4)'
    )
    args = parser.parse_args()
#pipeline
    changeDetection(
        args.input_folder,
        args.output_folder,
        args.input_ext,
        args.output_ext,
        args.video_format
    )


if __name__ == "__main__":
    main()
