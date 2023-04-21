import os
import requests
from tqdm import tqdm
import argparse


def download_weights(url, file_save_name):
    data_dir = "src/infra/checkpoints/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Download the file if not present.
    if not os.path.exists(os.path.join(data_dir, file_save_name)):
        print(f"Downloading {file_save_name}")
        file = requests.get(url, stream=True)
        total_size = int(file.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True
        )
        with open(os.path.join(data_dir, file_save_name), 'wb') as f:
            for data in file.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    else:
        print('File already present')


def parse_meta_file():
    weigths = open("src/infra/scripts/weights.txt", "r")
    data = weigths.read()
    data_into_list = data.split('\n')
    return data_into_list


def get_model(model_name):
    # Get the list containing all the weight file download URLs.
    weights_list = parse_meta_file()

    download_url = None
    for weights in weights_list:
        if model_name == weights.split('/')[-2]:
            print(f"Founds weights: {weights}\n")
            download_url = weights
            break

    assert download_url != None, f"{model_name} weight file not found!!!"

    # Download the checkpoint file.
    download_weights(
        url=download_url,
        file_save_name=download_url.split('/')[-1]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights', default='faster_rcnn_r50_fpn_1x_coco',
        help='weight file name'
    )
    args = vars(parser.parse_args())

    get_model(args['weights'])
