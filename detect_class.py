import argparse 

import time

#import serial

import torch.backends.cudnn as cudnn

#from utils.datasets import *
from utils.dataloaders import *
#from utils.utils import *
from utils import *

#serial_port = serial.Serial(
#    port="/dev/ttyTHS0",
#    baudrate=115200,
#)

class Yolov5Detection():
    def __init__(self,source):
        self.output = 'inference/output'  # 原始图片路径
        self.source = source #'inference/images'  # 矩阵转换后的图片保存路径
        self.weights = 'runs/train/exp4/weights/best.pt'  # 模型路径
        self.save_txt = False
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.5
        self.classes = None
        self.agnostic_nms = True
        self.fourcc = 'mp4v'
        self.view_img = False
        self.save_img = False
        self.augment = True
        self.project = "runs/detect"
        self.name = "exp"
        self.exist_ok = True
        self.device = torch_utils.select_device('')

    def detect(self):
        webcam = self.source == '0' or self.source.startswith('rtmp') or self.source.startswith('http') or self.source.endswith('.txt')

        if os.path.exists(self.output):
            shutil.rmtree(self.output)  # delete output folder
        os.makedirs(self.output)  # make new output folder
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = torch.load(self.weights, map_location=self.device)['model'].float()  # load to FP32
        model.to(self.device).eval()
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            modelc.to(self.device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            if self.source.endswith('.txt'):
                self.save_img = True
                dataset = LoadImages(self.source, img_size=self.img_size)
            else:
                self.view_img = True
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(self.source, img_size=self.img_size)
        else:
            self.save_img = True
            dataset = LoadImages(self.source, img_size=self.img_size)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    if self.source.endswith('.txt'):
                        p, s, im0 = path, '', im0s
                    else:
                        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(self.output) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1]:#.unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
			# add your upload code here ; and the object name is names[int(cls)]
                        print(int(cls))
                        a=int(cls)
                        a=a+1
                        tx=str('\r\n%s'%a)
                        time.sleep(0.05)
                        #serial_port.write(tx.encode("utf-8"))
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if self.save_img or self.view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if self.view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if self.save_txt or self.save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + self.output)
            if platform == 'darwin':  # MacOS
                os.system('open ' + self.save_path)

        print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == "__main__":
    detect = Yolov5Detection("0")
    detect.detect()
