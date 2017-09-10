# execute: python video.py run1
# Creates a video based on images found in the run1 directory. 
# The name of the video will be the name of the directory followed by '.mp4', 
# so, in this case the video will be run1.mp4.
# Optionally, one can specify the FPS (frames per second) of the video:
#  python video.py run1 --fps 48
# The video will run at 48 FPS. The default FPS is 60.

from moviepy.editor import ImageSequenceClip
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
