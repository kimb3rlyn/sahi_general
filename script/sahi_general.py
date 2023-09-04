import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from typing import List, Tuple
from typing_extensions import TypedDict

import torch
import numpy as np

from sahi.slicing import slice_image
from sahi.predict import POSTPROCESS_NAME_TO_CLASS
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

logger = logging.getLogger(__name__)

"""Detection: Dict[Detection class name, Detection confidence, LTRBWH bounding box]"""
DetectionType = TypedDict('Detection', {'label': str, 'confidence': float, 'l': int, 't': int, 'r': int, 'b': int, 'w': int, 'h': int})

class DetectionModel(ABC):
    @abstractmethod
    def detect(self, all_images: List[np.ndarray], classes: List[str] = None) -> List[List[DetectionType]]:
        """Performs detections."""
        raise NotImplementedError

@dataclass
class SahiGeneral(DetectionModel):
    """
    Args:
        model :
            any model
        sahi_image_height_threshold: int
            If image exceed this height, sahi will be performed on it.
            Defaulted to 900
        sahi_image_width_threshold: int
            If image exceed this width, sahi will be performed on it.
            Defaulted to 900
        sahi_slice_height: int
            Sliced image height.
            Defaulted to 512
        sahi_slice_width: int
            Sliced image width.
            Defaulted to 512
        sahi_overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels).
            Default to '0.2'.
        sahi_overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels).
            Default to '0.2'.
        sahi_postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
            Defaulted to "GREEDYNMM"
        sahi_postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
            Defaulted to "IOS"
        sahi_postprocess_match_threshold: float
            Sliced predictions having higher iou/ios than postprocess_match_threshold will be postprocessed after sliced prediction.
            Defaulted to 0.5
        sahi_postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
            Defaulted to True
        full_frame_detection: bool
            If True, additional detection will be done on the full frame.
            Defaulted to True
    """
    model: object = None
    sahi_image_height_threshold: int = 900
    sahi_image_width_threshold: int = 900
    sahi_slice_height: int = 512
    sahi_slice_width: int = 512
    sahi_overlap_height_ratio: float = 0.2
    sahi_overlap_width_ratio: float = 0.2
    sahi_postprocess_type: str = "GREEDYNMM"
    sahi_postprocess_match_metric: str = "IOS"
    sahi_postprocess_match_threshold: float = 0.5
    sahi_postprocess_class_agnostic: bool = True
    full_frame_detection: bool = True

    def __post_init__(self):
        assert (
            self.model is not None
        ), f"Model cannot be None"

        assert(
            self.sahi_postprocess_type.upper() in ["GREEDYNMM","NMM","NMS","LSNMS"]
        ), f"{self.sahi_postprocess_type} is not a valid option for the post processing method"

        self.sahi_postprocess_type = self.sahi_postprocess_type.upper()

        assert(
            self.sahi_postprocess_match_metric.upper() in ["IOU", "IOS"]
        ), f"{self.sahi_postprocess_match_metric} is not a valid option for the post processing match metric"

        self.sahi_postprocess_type = self.sahi_postprocess_type.upper()

        logger.info('SahiGeneral warmed up')

    @torch.no_grad()
    def detect(self, all_images: List[np.ndarray], classes: List[str] = None) -> List[List[DetectionType]]:
        """
        Args:
            list of images (np.ndarray): an image of shape (H, W, C) (in BGR order).
            classes (List[str]) : classes to focus on (optional).
        Returns:
            predictions (dict): the output of the model
        """
        # Images that are below the threshold will be ignored for SAHI and undergo normal detection
        list_of_non_sahi_imgs = []
        list_of_sahi_imgs = []
        list_of_sahi_idx = []
        for i, img in enumerate(all_images):
            if img.shape[0] > self.sahi_image_height_threshold or img.shape[1] > self.sahi_image_width_threshold:
                list_of_sahi_imgs.append(img)
                list_of_sahi_idx.append(i)
            else:
                list_of_non_sahi_imgs.append(img)

        if self.full_frame_detection:
            # Do detections on both SAHI and non-SAHI images
            images_to_detect = all_images
        else:
            # Do detection on full frame for only non-SAHI images
            images_to_detect = list_of_non_sahi_imgs

        all_detections = self.model.get_detections_dict(images_to_detect, classes=classes)
        if all_detections is None:
            all_detections = [[]] * len(images_to_detect)

        # batch up SAHI detection
        sahi_predictions, all_shift_amount, all_full_shape = self._sahi_detection_batch(list_of_sahi_imgs, classes)

        # combine SAHI detections with full image detection
        all_sahi_results = []
        for i, idx in enumerate(list_of_sahi_idx):
            if self.full_frame_detection:
                full_frame_predictions = all_detections[idx]
            else:
                full_frame_predictions = []
            sahi_detection_results = self._detect_sahi(full_frame_predictions, sahi_predictions[i], all_shift_amount[i], all_full_shape[i])
            all_sahi_results.append(sahi_detection_results)

        # convert detections from SAHI format to dictionary format
        for i, idx in enumerate(list_of_sahi_idx):
            img_result = []
            for sahi_result in all_sahi_results[i]:
                for obj_det in sahi_result:
                    l = int(obj_det.bbox.minx)
                    t = int(obj_det.bbox.miny)
                    r = int(obj_det.bbox.maxx)
                    b = int(obj_det.bbox.maxy)
                    w = r - l
                    h = b - t
                    score = obj_det.score.value
                    class_name = obj_det.category.name
                    result = {'label': class_name, 'confidence': score, 't': t, 'l': l, 'b': b, 'r': r, 'w': w, 'h': h}
                    img_result.append(result)
            if self.full_frame_detection:
                # replace result with the combination of both SAHI and non-SAHI detections
                all_detections[idx] = img_result
            else:
                # insert SAHI detections into the correct idx
                all_detections.insert(idx, img_result)
        return all_detections

    @torch.no_grad()
    def _sahi_detection_batch(self, sahi_imgs, classes=None):
        all_sahi_images = []
        all_shift_amount_list = []
        all_full_shape_list = []
        for img in sahi_imgs:
            slice_image_result = slice_image(
                image=img,
                slice_height=self.sahi_slice_height,
                slice_width=self.sahi_slice_width,
                overlap_height_ratio=self.sahi_overlap_height_ratio,
                overlap_width_ratio=self.sahi_overlap_width_ratio
            )

            shift_amount_list = []
            full_shape_list = []
            for sliced_image in slice_image_result.sliced_image_list:
                all_sahi_images.append(sliced_image.image)
                shift_amount_list.append(sliced_image.starting_pixel)
                full_shape_list.append([slice_image_result.original_image_height, slice_image_result.original_image_width])
            all_shift_amount_list.append(fix_shift_amount_list(shift_amount_list))
            all_full_shape_list.append(fix_full_shape_list(full_shape_list))
            logger.debug(f'Number of sahi slices for 1 image: {len(shift_amount_list)}')

        all_detections = self.model.get_detections_dict(all_sahi_images, classes=classes)
        if all_detections is not None:
            all_sahi_predictions = iter(all_detections)
        else:
            all_sahi_predictions = [[]] * len(all_sahi_images)
        sahi_predictions = [list(islice(all_sahi_predictions, len(fs_list))) for fs_list in all_full_shape_list]
        return sahi_predictions, all_shift_amount_list, all_full_shape_list

    def _detect_sahi(self, non_sahi_predictions, sahi_predictions, shift_amount_list, full_shape_list):
        object_prediction_list = []

        # non-SAHI detections
        for prediction in non_sahi_predictions:
            object_prediction = self._convert_dict_to_sahi_format(prediction, [0, 0], None)
            if object_prediction is not None:
                object_prediction_list.append(object_prediction)

        # SAHI detections
        for idx, predictions in enumerate(sahi_predictions):
            shift_amount = shift_amount_list[idx]
            full_shape = None if full_shape_list is None else full_shape_list[idx]

            for prediction in predictions:
                object_prediction = self._convert_dict_to_sahi_format(prediction, shift_amount, full_shape)
                if object_prediction is not None:
                    object_prediction_list.append(object_prediction.get_shifted_object_prediction())

        # postprocess both SAHI and non-SAHI detections
        return [self._postprocess(object_prediction_list)]

    def _convert_dict_to_sahi_format(self, prediction, shift_amount, full_shape):
        x1 = int(prediction['l'])
        y1 = int(prediction['t'])
        x2 = int(prediction['r'])
        y2 = int(prediction['b'])
        bbox = [x1, y1, x2, y2]
        score = prediction['confidence']
        category_name = prediction['label']

        # fix negative box coords
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = max(0, bbox[2])
        bbox[3] = max(0, bbox[3])

        # fix out of image box coords
        if full_shape is not None:
            bbox[0] = min(self.sahi_slice_width, bbox[0])
            bbox[1] = min(self.sahi_slice_height, bbox[1])
            bbox[2] = min(self.sahi_slice_width, bbox[2])
            bbox[3] = min(self.sahi_slice_height, bbox[3])

        # ignore invalid predictions
        if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
            logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
            return None
        return ObjectPrediction(
                bbox=bbox,
                score=score,
                category_id = self.model.classname_to_idx(category_name),
                bool_mask=None,
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )

    def _postprocess(self, object_prediction_list):
        if len(object_prediction_list) == 1:
            return object_prediction_list
        elif object_prediction_list:
            postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[self.sahi_postprocess_type]
            postprocess = postprocess_constructor(
                match_threshold=self.sahi_postprocess_match_threshold,
                match_metric=self.sahi_postprocess_match_metric,
                class_agnostic=self.sahi_postprocess_class_agnostic,
            )
            return postprocess(object_prediction_list)
        else:
            return []
