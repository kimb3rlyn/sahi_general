from importlib_resources import files
from pathlib import Path
from time import perf_counter

import cv2
import torch

from script.sahi_general import SahiGeneral
from yolov7.yolov7 import YOLOv7


imgpath = Path('test.jpg')
if not imgpath.is_file():
    raise AssertionError(f'{str(imgpath)} not found')

output_folder = Path('inference')
output_folder.mkdir(parents=True, exist_ok=True)


yolov7 = YOLOv7(
    weights=files('yolov7').joinpath('weights/yolov7_state.pt'),
    cfg=files('yolov7').joinpath('cfg/deploy/yolov7.yaml'),
    bgr=True,
    device='cuda',
    model_image_size=640,
    max_batch_size=16,
    half=True,
    same_size=True,
    conf_thresh=0.25,
    trace=False,
    cudnn_benchmark=False,
)

classes = ['person', 'car']

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
bs = 8
imgs = [img for _ in range(bs)]

torch.cuda.synchronize()
tic = perf_counter()
detections = sahi_general.detect(imgs, classes)

torch.cuda.synchronize()
dur = perf_counter() - tic

print(f'Time taken: {(dur*1000):0.2f}ms')

draw_frame = img.copy()

for det in detections[0]:
    l = det['l']
    t = det['t']
    r = det['r']
    b = det['b']
    classname = det['label']
    cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
    cv2.putText(draw_frame, classname, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

output_path = output_folder / 'test_out.jpg'
cv2.imwrite(str(output_path), draw_frame)
