from importlib_resources import files
from pathlib import Path
from time import perf_counter

import cv2
import torch

from sahi_general import SahiGeneral
from yolov7.yolov7 import YOLOv7


imgpath = Path('test.jpg')
if not imgpath.is_file():
    raise AssertionError(f'{str(imgpath)} not found')

output_folder = 'inference'
Path(output_folder).mkdir(parents=True, exist_ok=True)


yolov7 = YOLOv7(
    weights='yolov7_state.pt',
    cfg='./yolov7.yaml',
    bgr=True,
    gpu_device=0,
    model_image_size=640,
    max_batch_size=16,
    half=True,
    same_size=True,
    conf_thresh=0.25,
    trace=False,
    cudnn_benchmark=False,
)


'''
    SAHI library needs to be installed 
    Model needs to have classname_to_idx function and get_detections_dict function
    classname_to_idx : int
        class index of the classname given 
    get_detections_dict : List[dict]
        list of detections for each frame with keys: label, confidence, t, l, b, r, w, h
'''
sahi_general = SahiGeneral(model=yolov7)

img = cv2.imread(str(imgpath))
bs = 64
imgs = [img for _ in range(bs)]

dur = 0

torch.cuda.synchronize()
tic = perf_counter()
'''
    If one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
    else if a list of ndarray given, this return a list (batch) containing the former as the elements.
'''
dets = sahi_general.detect(imgs)

torch.cuda.synchronize()
toc = perf_counter()
dur += toc - tic


print(f'Average time taken: {(dur*1000):0.2f}ms')

draw_frame = img.copy()

for chip in dets:
    for det in chip:
        bb, score, class_ = det
        l, t, r, b = bb
        cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
        cv2.putText(draw_frame, class_, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

output_path = Path(output_folder) / 'test_out.jpg'
cv2.imwrite(str(output_path), draw_frame)
