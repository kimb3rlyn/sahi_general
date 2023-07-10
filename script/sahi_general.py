import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import torch
import numpy as np

from sahi.slicing import slice_image
from sahi.predict import POSTPROCESS_NAME_TO_CLASS
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

logger = logging.getLogger(__name__)

"""Bounding box LTRB values"""
BoundingBoxType = Tuple[float, float, float, float]

"""Detection: Tuple[LTRB bounding box, Detection confidence, Detection class name]"""
DetectionType = Tuple[BoundingBoxType, float, str]

class DetectionModel(ABC):
    @abstractmethod
    def detect(self, list_of_imgs: List[np.ndarray]) -> List[List[DetectionType]]:
        """Performs detections. This method should call self._preprocess() and self._postprocess()"""
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

        print('SAHI detection warmed up')

    @torch.no_grad()
    def detect(self, list_of_imgs: List[np.ndarray], classes=None) -> List[List[DetectionType]]:
        """
        Args:
            list of images (np.ndarray): an image of shape (H, W, C) (in BGR order).
            classes (List[str]) : classes to focus on (optional).
        Returns:
            predictions (dict): the output of the model
        """
        # split image into 2 array, 1 for sahi and 1 for batch
        # another array will be for store how to merge them back
        list_of_non_sahi_imgs = []
        list_of_sahi_imgs = []
        list_of_sahi_idx = []

        # In the case of irregular image size, the images that are below the threshold will be ignore for SAHI and undergo normal detection
        for i, img in enumerate(list_of_imgs):
            if img.shape[0] > self.sahi_image_height_threshold or img.shape[1] > self.sahi_image_width_threshold:
                list_of_sahi_imgs.append(img)
                list_of_sahi_idx.append(i)
            else:
                list_of_non_sahi_imgs.append(img)

        # SAHI detection
        list_sahi_detection_results = []
        for sahi_img in list_of_sahi_imgs:
            # send to inference each of sahi images
            sahi_detection_results = self._detect_sahi(sahi_img, classes)
            list_sahi_detection_results.append(sahi_detection_results)

        # Non-SAHI detection
        detection_results = self.model.get_detections_dict(list_of_non_sahi_imgs, classes=classes)
        if detection_results is None:
            detection_results = [[]]

        # merge sahi det results with original list result
        exit_counter = 0
        for i, idx in enumerate(list_of_sahi_idx):
            img_result_list = []
            for sahi_slice_result in list_sahi_detection_results[i]:
                if len(sahi_slice_result) > 0:
                    for obj_det in sahi_slice_result:
                        l = obj_det.bbox.minx
                        t = obj_det.bbox.miny
                        r = obj_det.bbox.maxx
                        b = obj_det.bbox.maxy
                        w = r - l
                        h = b - t
                        score = obj_det.score.value
                        class_name = obj_det.category.name
                        result = {'label': class_name, 'confidence': score, 't': t, 'l': l, 'b': b, 'r': r, 'w': w, 'h': h}
                        img_result_list.append(result)
            detection_results.insert(idx, img_result_list)
        return detection_results

    @torch.no_grad()
    def _detect_sahi(self, img, classes):

        slice_image_result = slice_image(
            image=img,
            slice_height=self.sahi_slice_height,
            slice_width=self.sahi_slice_width,
            overlap_height_ratio=self.sahi_overlap_height_ratio,
            overlap_width_ratio=self.sahi_overlap_width_ratio
        )

        postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[self.sahi_postprocess_type]
        postprocess = postprocess_constructor(
            match_threshold=self.sahi_postprocess_match_threshold,
            match_metric=self.sahi_postprocess_match_metric,
            class_agnostic=self.sahi_postprocess_class_agnostic,
        )

        # create prediction list
        shift_amount_list = []
        list_of_imgs = []
        full_shape_list = []
        for sliced_image in slice_image_result.sliced_image_list:
            list_of_imgs.append(sliced_image.image)
            shift_amount_list.append(sliced_image.starting_pixel)
            full_shape_list.append(
                [slice_image_result.original_image_height, slice_image_result.original_image_width])

        original_detection_results = self.model.get_detections_dict([img], classes=classes)[0]

        object_prediction_list = []
        for prediction in original_detection_results:
            x1 = int(prediction['l'])
            y1 = int(prediction['t'])
            x2 = int(prediction['r'])
            y2 = int(prediction['b'])
            bbox = [x1, y1, x2, y2]
            score = prediction['confidence']
            category_name = prediction['label']

            # ignore invalid predictions
            if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                logger.warning(
                    f"ignoring invalid prediction with bbox: {bbox}")
                continue
            object_prediction = ObjectPrediction(
                bbox=bbox,
                score=score,
                category_id = self.model.classname_to_idx(category_name),
                bool_mask=None,
                category_name=category_name,
                shift_amount=[0, 0],
                full_shape=None,
            )
            object_prediction_list.append(object_prediction)

        original_predictions = self.model.get_detections_dict(list_of_imgs, classes=classes)
  
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            
            # process predictions
            for prediction in image_predictions:
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
                    logger.warning(
                        f"ignoring invalid prediction with bbox: {bbox}")
                    continue
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    score=score,
                    category_id = self.model.classname_to_idx(category_name),
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(
                    object_prediction.get_shifted_object_prediction())

        # combine predictions
        object_prediction_list = postprocess(object_prediction_list)
        
        return [object_prediction_list]
