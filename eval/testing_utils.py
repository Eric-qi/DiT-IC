import argparse

def parse_args_testing(input_args=None):

    parser = argparse.ArgumentParser()
    # testing setting
    parser.add_argument("--config_path", type=str, default='/data/Kodak/')
    parser.add_argument("--codec_path", type=str, default=None)
    # testing images
    parser.add_argument("--img_path", type=str, default='/data/Kodak/')

    # output path
    parser.add_argument("--rec_path", type=str, default='/output/rec/')
    parser.add_argument("--bin_path", type=str, default='/output/bin/')

    # testing details
    parser.add_argument("--seed", type=int, default=903, help="A seed for reproducible training.")
    parser.add_argument("--use_ema", action="store_true",)
    parser.add_argument("--save_img", action="store_true",)
    parser.add_argument("--use_merge", action="store_true",)
    parser.add_argument("--entropy_estimation", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
