from PIL import Image, ImageDraw, ImageFont


def label_incorrect_predictions(
    image_matrix,
    predictions,
    true_labels,
    label_names,
    img_size,
    no_label_indicator,
    font_size=30,
    padding=5,
):

    print("labeling incorrect predictions")
    draw = ImageDraw.Draw(image_matrix)
    font = ImageFont.truetype("arial.ttf", font_size)

    num_columns = image_matrix.width // img_size
    num_rows = image_matrix.height // img_size

    # add annotations for incorrectly predicted labeled images
    for col in range(num_columns):
        for row in range(num_rows):
            idx = col * num_rows + row
            pred = predictions[idx]
            true_label = true_labels[idx]

            if pred != true_label and true_label != no_label_indicator:
                # calculate position of the image
                x0, y0 = col * img_size, row * img_size

                # draw correct label name on the image
                label_name = f"{label_names[true_label]}"
                draw.text(
                    (x0 + padding, y0 + padding), label_name, fill="blue", font=font
                )

    return image_matrix


def add_label_names(image_matrix, label_names, label_name_height=50, font_size=25):
    # create a new image with space on top for the label names
    new_image_matrix = Image.new(
        mode="RGB",
        size=(image_matrix.width, image_matrix.height + label_name_height),
        color="white",
    )
    new_image_matrix.paste(image_matrix, box=(0, label_name_height))

    draw = ImageDraw.Draw(new_image_matrix)
    font = ImageFont.truetype("arial.ttf", font_size)
    img_width = image_matrix.width

    # calculate the maximum text height
    max_text_height = max(
        draw.textbbox((0, 0), label_name, font=font)[3] for label_name in label_names
    )

    # add names to the top of the image
    for idx, label_name in enumerate(label_names):
        # calculate bounding box of the text
        text_bbox = draw.textbbox((0, 0), label_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        # calculate position for the text
        text_x = (
            idx * (img_width // len(label_names))
            + (img_width // len(label_names) - text_width) // 2
        )
        text_y = (label_name_height - max_text_height) // 2

        draw.text((text_x, text_y), label_name, fill="black", font=font)

    return new_image_matrix
