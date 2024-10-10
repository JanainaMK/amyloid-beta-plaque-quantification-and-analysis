import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.categorisation.cli import parse_config
from src.categorisation.plaques_dataset import PlaquesDataset


def create_image_matrices(
    dataloader,
    n_samples: int,
    start_sample_num: int,
    n_cols: int,
    n_rows: int,
    img_size: int,
    save_dir: str,
    samples_save_dir: str,
    **kwargs,
):
    n_image_matrices = n_samples
    samples_save_path = save_dir + "/" + samples_save_dir
    os.makedirs(samples_save_path, exist_ok=True)
    print("number of samples:", n_image_matrices)
    print("number of columns for each sample:", n_cols)
    print("number of rows for each sample:", n_rows, "\n")
    curr_image_matrix = np.zeros((n_rows * img_size, n_cols * img_size, 3))

    image_matrix_idx = 0
    row_idx = 0
    col_idx = 0

    # indicates at what sample number to start saving images
    n_image_matrices += start_sample_num

    for batch in dataloader:
        if n_image_matrices == 0:
            return
        imgs, _ = batch
        imgs = np.transpose(imgs, (0, 2, 3, 1))

        for img in imgs:
            # if number of rows is reached, move on to next column
            if row_idx >= n_rows:
                row_idx = 0
                col_idx += 1

            # if number of columns is reached, move on to next image matrix
            if col_idx >= n_cols:

                if image_matrix_idx >= start_sample_num:
                    image_save_path = (
                        f"{samples_save_path}/sample_{image_matrix_idx}.png"
                    )
                    curr_image_matrix = curr_image_matrix * 255.0
                    curr_image_matrix = curr_image_matrix.astype(np.uint8)
                    curr_image_matrix = Image.fromarray(curr_image_matrix)

                    # add annotations for column/row numbers
                    font_size = 30
                    font = ImageFont.truetype("arial.ttf", font_size)
                    draw = ImageDraw.Draw(curr_image_matrix)
                    for i in range(n_rows):
                        draw.text(
                            (0, i * img_size + img_size // 2),
                            str(i),
                            font=font,
                            fill="black",
                        )
                    for j in range(n_cols):
                        draw.text(
                            (j * img_size + img_size // 2, 0),
                            str(j),
                            font=font,
                            fill="black",
                        )

                    curr_image_matrix.save(image_save_path)
                    print(f"saved sample_{image_matrix_idx}")

                image_matrix_idx += 1

                if image_matrix_idx >= n_image_matrices:
                    # if sample number is reached, end method
                    return
                else:
                    # if sample number is not reached, continue creating next image matrix sample
                    row_idx = 0
                    col_idx = 0
                    if image_matrix_idx >= start_sample_num:
                        curr_image_matrix = np.zeros(
                            (n_rows * img_size, n_cols * img_size, 3)
                        )

            # start saving images with relevant start sample number
            if image_matrix_idx >= start_sample_num:
                # save image in image matrix
                curr_image_matrix[
                    row_idx * img_size : row_idx * img_size + img_size,
                    col_idx * img_size : col_idx * img_size + img_size,
                ] = img
            row_idx += 1

    print("saved samples")


def main():
    config = parse_config()

    datamodule = PlaquesDataset(
        **config["data_params"], pin_memory=len(config["run_params"]["gpus"]) != 0
    )
    datamodule.setup()

    dataloader = datamodule.train_dataloader()
    create_image_matrices(dataloader=dataloader, **config["samples_param"])


if __name__ == "__main__":
    main()
