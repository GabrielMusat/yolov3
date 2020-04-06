import argparse
from models import *
from utils.datasets import *
from utils.utils import *
import cv2
import os


def detect():
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size
    source, weights, half = opt.source, opt.weights, opt.half
    source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    model = Darknet(opt.cfg, img_size)

    attempt_download(weights)
    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        load_darknet_weights(model, weights)

    model.to(device).eval()

    half = half and device.type != 'cpu'
    if half:
        model.half()

    dataset = LoadImages(source, img_size=img_size)
    for path, img, im0s, vid_cap in dataset:
        t0 = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0].float() if half else model(img)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                cont = 0
                for *xyxy, conf, cls in det:
                    file = path.split("/")[-1]
                    name = "_".join(file.split('_')[:2])
                    if not os.path.isdir(f'{opt.output}/{name}'): os.mkdir(f'{opt.output}/{name}')
                    cv2.imwrite(f'{opt.output}/{name}/box{cont}-{file}', im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :])
                    cont += 1
        print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg_16/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--output', type=str, default='pieces', help='output path')
    parser.add_argument('--weights', type=str, default='weights_16/last.pt', help='weights path')
    parser.add_argument('--source', type=str, default='samples', help='source')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    if not os.path.isdir(opt.output): os.mkdir(opt.output)

    with torch.no_grad():
        detect()
