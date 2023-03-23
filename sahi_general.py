import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from sahi.slicing import slice_image
from sahi.annotation import BoundingBox
from sahi.predict import POSTPROCESS_NAME_TO_CLASS
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import check_requirements, is_available
from sahi.utils.torch import is_torch_cuda_available
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

logger = logging.getLogger(__name__)

"""Bounding box LTRB values"""
BoundingBoxType = Tuple[float, float, float, float]

"""Detection: Tuple[LTRB bounding box, Detection confidence, Detection class name]"""
DetectionType = Tuple[BoundingBoxType, float, str]

def batch(iterable, bs=1):
    """Yields iterable in batches of size bs"""
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx: min(ndx + bs, l)]

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return 


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
        image_size: int
            Inference input size.
        bgr : bool 
            If True, images are in BGR. Defaulted to False
        batch_size : int 
            Batch size of the  model 
        trace :
            If True, model is trace. Defaulted to False
        flip_channels :
            If True, flip_channels is applied. Defaulted to True

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
            Defaulted to "IOU"
        sahi_postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be postprocessed after sliced prediction.
            Defaulted to 0.5
        sahi_postprocess_class_agnostic: bool 
            If True, postprocess will ignore category ids.
            Defaulted to True
    """
    model: object = None
    image_size: Optional[int] = 1920  # multiple of 64
    bgr: bool = True
    batch_size: int = 1
    trace: bool = False
    half: Optional[bool] = None
    flip_channels: bool = True

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
    def detect(self, list_of_imgs: List[np.ndarray]) -> List[List[DetectionType]]:
        """
        Args:
            list of images (np.ndarray): an image of shape (H, W, C) (in BGR order).
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
            # send to inference each of sahi images in a batch
            sahi_detection_results = self._detect_sahi(sahi_img)
            list_sahi_detection_results.append(sahi_detection_results)

        # Normal detection
        detection_results = []
        if len(list_of_non_sahi_imgs) > 0:
            predictions = self.model.get_detections_dict(list_of_imgs)[0]
            detection_results = predictions

        # merge sahi det results  with original list result
        exit_counter = 0
        for i, idx in enumerate(list_of_sahi_idx):
            img_result_list = []
            for sahi_slice_result in list_sahi_detection_results[i]:
                if len(sahi_slice_result) > 0:
                    for obj_det in sahi_slice_result:
                        ltrb = [
                            obj_det.bbox.minx,
                            obj_det.bbox.miny,
                            obj_det.bbox.maxx,
                            obj_det.bbox.maxy,
                        ]
                        score = obj_det.score.value
                        class_name = obj_det.category.name
                        img_result_list.append([ltrb, score, class_name])
            detection_results.insert(idx, img_result_list)
        return detection_results

    @torch.no_grad()
    def _detect_sahi(self, img):

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

        # prepare batches
        original_predictions = []
        batches = batch(list_of_imgs, bs=self.batch_size)
        for b in batches:
            # predict each batch
            original_preds = self.model.get_detections_dict(b)[0]
            original_predictions.extend([original_preds])
  
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        object_prediction_list = []


        # needed format for original_predictions is xyxy score classid
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            
            # process predictions
            for prediction in image_predictions_in_xyxy_format:
                
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
