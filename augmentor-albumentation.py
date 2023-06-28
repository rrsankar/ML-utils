import argparse
import os
from copy import deepcopy
from datetime import datetime

import albumentations as alb
import cv2
import numpy as np


class ImageAugmentation:
    """
    Class to perform image augmentations.
    """

    def __init__(self,
                 images_path: str,
                 labels_path: str,
                 augmented_images_path: str,
                 augmented_labels_path: str,
                 augmented_images_with_bbox_path: str,
                 background_images_path: str,
                 image_format: str,
                 required_augmentations: list = None):

        self.images_path = images_path
        self.labels_path = labels_path
        self.augmented_images_path = augmented_images_path
        self.augmented_labels_path = augmented_labels_path
        self.augmented_images_with_bbox_path = augmented_images_with_bbox_path
        self.background_images_path = background_images_path
        self.image_format = image_format
        self.required_augmentations = required_augmentations

        self.defined_augmentations = {
            # Pixel-level
            "ToGray": alb.ToGray(p=1),
            "Blur": alb.Blur(p=1),
            "MedianBlur": alb.MedianBlur(p=1),
            "MotionBlur": alb.MotionBlur(p=1),
            "ChannelDropout": alb.ChannelDropout(p=1),
            "ChannelShuffle": alb.ChannelShuffle(p=1),
            "RandomBrightness": alb.RandomBrightness(p=1),
            "RandomContrast": alb.RandomContrast(p=1),
            "HueSaturationValue": alb.HueSaturationValue(p=1),
            "MultiplicativeNoise": alb.MultiplicativeNoise(p=1),
            "RGBShift": alb.RGBShift(p=1),
            "RandomBrightnessContrast": alb.RandomBrightnessContrast(p=1),
            # Spatial-level
            "VerticalFlip": alb.VerticalFlip(p=1),
            "HorizontalFlip": alb.HorizontalFlip(p=1),
            "Rotate": alb.Rotate(limit=(-90, 90), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1),
            "ShiftScaleRotate": alb.ShiftScaleRotate(p=1),
            "RandomSizedCrop": alb.RandomSizedCrop(height=1440, width=2048, min_max_height=(1080, 1080), p=1),
            "RandomResizedCrop": alb.RandomResizedCrop(height=1440, width=2048, p=1),
            "RandomSizedBBoxSafeCrop": alb.RandomSizedBBoxSafeCrop(height=1440, width=2048, p=1),
        }

        # Check if all input augmentations is present in our defined augmentations.
        for inp_iter in self.required_augmentations:
            if inp_iter not in self.defined_augmentations.keys():
                raise ValueError(f"Requested augmentation '{inp_iter}' is not defined in this program.")

    def _draw_bbox(self, image, bboxes: list):
        line_color = (255, 0, 0)
        line_thickness = 2
        image_height, image_width, _ = image.shape
        for coordinate in bboxes:
            yolo_x, yolo_y, yolo_w, yolo_h = coordinate
            left = int((yolo_x - yolo_w / 2) * image_width)
            right = int((yolo_x + yolo_w / 2) * image_width)
            top = int((yolo_y - yolo_h / 2) * image_height)
            bottom = int((yolo_y + yolo_h / 2) * image_height)
            if left < 0:
                left = 0
            if right > image_width - 1:
                right = image_width - 1
            if top < 0:
                top = 0
            if bottom > image_height - 1:
                bottom = image_height - 1
            try:
                image = cv2.rectangle(image, (left, top), (right, bottom), line_color, line_thickness)
            except:
                image = cv2.rectangle(np.ascontiguousarray(image, dtype=np.uint8), (left, top), (right, bottom), line_color, line_thickness)
        return image

    def _perform_augmentation(self, augmentation: str, image_path: str, label_path: str):
        """
        Perform augmentation for a single image and its labels.
        """

        # Read an image with OpenCV and convert it to the RGB colorspace
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read corresponding labels and convert to required format.
        annotations, class_labels = [], []
        with open(label_path, "r") as f:
            annotation_lines = f.readlines()
        for idx in range(len(annotation_lines)):
            line_split = annotation_lines[idx].strip().split()
            class_labels.append(line_split[0])
            annotations.append([float(i) for i in line_split[1:5]])

        # Declare an augmentation pipeline
        transform = alb.Compose([
            self.defined_augmentations.get(augmentation)
        ], bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']))

        # Augment an image
        transformed = transform(image=image, bboxes=annotations, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        if (image.shape[1], image.shape[0]) != (transformed_image.shape[1], transformed_image.shape[0]):
            raise ValueError(f"{augmentation} will cause a size variation from (width,height)={(image.shape[1], image.shape[0])} "
                             f"to {(transformed_image.shape[1], transformed_image.shape[0])}.")

        return transformed_image, transformed_bboxes, transformed_class_labels

    def _save_image(self, image, image_name, save_path: str):

        # Create save directory if not exists.
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save image.
        cv2.imwrite(os.path.join(save_path, image_name), image)

    def _save_labels_in_yolo_format(self, labels: list, class_labels: list, file_name, save_path: str, background_image: bool = False):

        # Create save directory if not exists.
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if os.path.exists(os.path.join(save_path, file_name)):
            raise ValueError("Label already exist.")

        if not background_image:
            # Write lines as "class x,y,w,h" which is the yolo format
            for label_index in range(len(labels)):
                x,y,w,h = labels[label_index]
                with open(os.path.join(save_path, file_name), "a+") as f:
                    f.writelines(f"{class_labels[label_index]} {x} {y} {w} {h}\n")

        # If background image, create empty txt file.
        else:
            with open(os.path.join(save_path, file_name), "a") as f:
                pass

    def main(self):

        # Read directory
        dir_files = os.listdir(self.images_path)
        required_files = [i for i in dir_files if i.endswith(f'.{self.image_format}')]

        # Formulate chosen augmentations.
        if self.required_augmentations:
            chosen_augmentations = deepcopy(self.required_augmentations)
        else:
            chosen_augmentations = self.defined_augmentations.keys()

        # Loop through images and labels and perform chosen augmentations.
        for file_iter in required_files:

            image_path = os.path.join(self.images_path, file_iter)
            label_path = os.path.join(self.images_path, file_iter.replace(f".{self.image_format}", ".txt"))

            # Loop through all required augmentations and perform each aug on the current image & label and save it.
            for augmentation_iter in chosen_augmentations:

                print(f"{file_iter} - {augmentation_iter}\t*processing*")

                # Augmentation
                transformed_image, transformed_bboxes, transformed_class_labels = self._perform_augmentation(
                    augmentation=augmentation_iter,
                    image_path=image_path,
                    label_path=label_path
                )

                # define the final file name.
                final_file_name = f"{file_iter.split('.')[0]}_{augmentation_iter}"

                if transformed_bboxes:

                    # Save augmented images and labels.
                    self._save_labels_in_yolo_format(
                        labels=transformed_bboxes,
                        class_labels=transformed_class_labels,
                        file_name=final_file_name+".txt",
                        save_path=self.augmented_labels_path
                    )
                    self._save_image(
                        image=transformed_image,
                        image_name=final_file_name+f".{self.image_format}",
                        save_path=self.augmented_images_path
                    )

                    if self.augmented_images_with_bbox_path:

                        # Raise error if path doesn't exist.
                        if not os.path.exists(self.augmented_images_with_bbox_path):
                            raise ValueError(f"Path {self.augmented_images_with_bbox_path} doesn't exist.")

                        # Draw bbox and save images for reference separately.
                        aug_img_with_bbox = self._draw_bbox(image=transformed_image, bboxes=transformed_bboxes)

                        # Save augmented image with bboxes.
                        self._save_image(image=aug_img_with_bbox,
                                         image_name="ref_"+final_file_name+f".{self.image_format}",
                                         save_path=self.augmented_images_with_bbox_path)

                # BACKGROUND IMAGES -> NO OBJECTS / NO BBOXES.
                else:

                    self._save_labels_in_yolo_format(
                        labels=transformed_bboxes,
                        class_labels=transformed_class_labels,
                        file_name=final_file_name+".txt",
                        save_path=self.background_images_path,
                        background_image=True
                    )

                    self._save_image(
                        image=transformed_image,
                        image_name=final_file_name+f".{self.image_format}",
                        save_path=self.background_images_path
                    )


if __name__ == "__main__":

    start_time = datetime.now()

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding arguments
    parser.add_argument("--images_path", type=str, help="Training images path.")
    parser.add_argument("--labels_path", type=str, help="Training labels path.")
    parser.add_argument("--augmented_images_path", type=str, help="Output augmented images path.")
    parser.add_argument("--augmented_labels_path", type=str, help="Output augmented labels path.")
    parser.add_argument("--augmented_images_with_bbox_path", default="", type=str, help="[Optional] Path to augmented images with bbox for reference.")
    parser.add_argument("--background_images_path", type=str, help="Path to store background images.")
    parser.add_argument("--image_format", type=str, help="Input and output image format.")
    parser.add_argument("--required_augmentations", type=str, nargs="*", help="List of all required augmentations")

    # Read arguments from command line
    args = parser.parse_args()

    if not all([args.images_path, args.labels_path, args.augmented_images_path, args.augmented_labels_path, args.background_images_path]):
        raise ValueError("Required field values missing.")

    for i in [args.images_path, args.labels_path, args.augmented_images_path, args.augmented_labels_path, args.background_images_path]:
        if not os.path.exists(i):
            raise ValueError(f"Path '{i}' doesn't exist.")

    # Create an object of the ImageAugmentation class.
    img_aug_obj = ImageAugmentation(
        images_path=args.images_path,
        labels_path=args.labels_path,
        augmented_images_path=args.augmented_images_path,
        augmented_labels_path=args.augmented_labels_path,
        augmented_images_with_bbox_path=args.augmented_images_with_bbox_path,
        background_images_path=args.background_images_path,
        image_format=args.image_format,
        required_augmentations=args.required_augmentations
    )

    img_aug_obj.main()

    print(f"\n\nTotal time: {(datetime.now()-start_time).seconds} seconds")


# PERFORM CERTAIN AUGMENTATIONS

# python albumentation_test.py \
# --images_path C:\Users\ramdo\Documents\pycharm-projects\RemoteCam-test\src\POC11-augmentation\data \
# --labels_path C:\Users\ramdo\Documents\pycharm-projects\RemoteCam-test\src\POC11-augmentation\data \
# --augmented_images_path C:\Users\ramdo\Documents\pycharm-projects\RemoteCam-test\src\POC11-augmentation\augmented \
# --augmented_labels_path C:\Users\ramdo\Documents\pycharm-projects\RemoteCam-test\src\POC11-augmentation\augmented \
# --augmented_images_with_bbox_path C:\Users\ramdo\Documents\pycharm-projects\RemoteCam-test\src\POC11-augmentation\augmented_bbox \
# --background_images_path C:\Users\ramdo\Documents\pycharm-projects\RemoteCam-test\src\POC11-augmentation\background \
# --image_format JPG \
# --required_augmentations ToGray Blur

# PERFORM ALL AUGMENTATION

# python albumentation_test.py \
# --images_path /home/ram/data \
# --labels_path /home/ram/data \
# --augmented_images_path /home/ram/augmented \
# --augmented_labels_path /home/ram/augmented \
# --augmented_images_with_bbox_path /home/ram/augmented-bbox \
# --background_images_path /home/ram/background_images \
# --image_format png \
