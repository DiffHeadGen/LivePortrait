import os
import os.path as osp
import shutil
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

from expdataloader import *
from expdataloader.utils import extract_all_frames, get_sub_dir


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def get_args():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(args)
    return args


def basename(filename):
    """a/b/c.jpg -> c"""
    return os.path.splitext(os.path.basename(filename))[0]


class LivePortraitLoader(RowDataLoader):
    def __init__(self, args=None, flags=[]):
        if args is None:
            args = get_args()
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)
        self.live_portrait_pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        self.args = args

        flags = ["LivePortrait"] + flags
        super().__init__("+".join(flags))

    def run_video(self, row):
        args = self.args
        args.source = row.source_img_path
        args.driving = row.target.video_path
        args.output_dir = row.output_dir
        self.live_portrait_pipeline.execute(args)
        ori_output_video_path = osp.join(args.output_dir, f"{basename(args.source)}--{basename(args.driving)}.mp4")
        shutil.copy(ori_output_video_path, row.output_video_path)
        row.output.human()


def no_relative_motion():
    args = get_args()
    args.flag_relative_motion = False
    loader = LivePortraitLoader(args, ["no_relative_motion"])
    loader.run_all()
    # row = loader.all_data_rows[21]
    # loader.run_video(row)


def no_stitching_and_crop():
    args = get_args()
    args.flag_stitching = False
    args.flag_do_crop = False
    args.flag_relative_motion = False
    loader = LivePortraitLoader(args, ["no_stitching", "no_crop", "no_relative_motion"])
    # loader.run_all()
    loader.test_20250218()


def main():
    no_stitching_and_crop()


if __name__ == "__main__":
    main()
