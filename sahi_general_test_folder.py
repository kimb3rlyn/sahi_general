from importlib_resources import files
from pathlib import Path
from time import perf_counter

import cv2
import torch

from script.sahi_general import SahiGeneral
from yolov7.yolov7 import YOLOv7

folder_path = Path('test_images_folder')
images_suffixes = ['.jpg', '.jpeg', '.png']

image_paths = []
for path in folder_path.iterdir():
    if path.suffix in images_suffixes:
        image_paths.append(path)

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

imgs = [cv2.imread(str(imgpath)) for imgpath in image_paths]

torch.cuda.synchronize()
tic = perf_counter()
detections = sahi_general.detect(imgs, classes)

torch.cuda.synchronize()
dur = perf_counter() - tic

print(f'Time taken: {(dur*1000):0.2f}ms')

for i, img in enumerate(imgs):
    draw_frame = img.copy()
    for det in detections[i]:
        l = det['l']
        t = det['t']
        r = det['r']
        b = det['b']
        classname = det['label']
        cv2.rectangle(draw_frame, (l, t), (r, b), (255, 255, 0), 1)
        cv2.putText(draw_frame, classname, (l, t-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

    output_path = output_folder / f'{image_paths[i].stem}_out.jpg'
    cv2.imwrite(str(output_path), draw_frame)
