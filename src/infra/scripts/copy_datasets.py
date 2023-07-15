import argparse
import shutil
import os

def copy(src, dst):
    if not os.path.isdir(dst):
        try:
            shutil.copytree(src, dst)
        except OSError as exc:
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(src, dst)
            else:
                raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets', default='sard,heridal', help='dataset names')
    parser.add_argument('-b', '--base-dir', default='src', help='base dir')
    parser.add_argument('-p', '--data-path', default='datasets',
                        help='datasets path')
    args = vars(parser.parse_args())
    for dataset in args["datasets"].split(","):
        copy(f"""{args["data_path"]}/{dataset}""", f"""{args["base_dir"]}/mmdetection/data/{dataset}""")
