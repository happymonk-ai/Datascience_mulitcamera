import asyncio
import errno
from itertools import count
from queue import Empty
from time import sleep
from turtle import shape
import nats
import json
import numpy as np 
from PIL import Image
import cv2
import torch
import os 
from multiprocessing import Process , Queue
from multiprocessing.managers import BaseManager

import os
import sys
from pathlib import Path
import threading

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from nanoid import generate
import logging



dict_frame = {}
frame = []
count_frame ={}
count = 0
length = []
device_list = []

async def detect(
        weights="./best_22Sep_2022.pt",  # model.pt path(s)
        source=ROOT,  # file/dir/URL/glob, 0 for webcam
        data="coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "./output" ,   # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        device_id = ROOT,
):
    source = str(source)
    person_count = []
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path("./output/"+device_id) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
        
    # Run inference
    frame_count = 0
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    global vehicle_count , license
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if names[int(c)] == "person" :
                        person_count.append(f"{n}")
                    if names[int(c)] == "vehicle":
                        vehicle_count.append(f"{n}")
            
               
                
                for c in det[:,-1]:
                    global personDid , count_person ,license_plate
                    # print(frame_count , "count 590")
                    if frame_count % 10 == 0: 
                        if names[int(c)]=="person":
                            print("person")
                        
                
                        
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                   
                    # print(save_img, "Save Image ")
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    frame_count += 1
                    
                                     
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        await asyncio.sleep(1)
        Process(target = LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')).start()
    
    
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        
async def  save_image(device_id): 
    global count
    try :  
        arr = np.ndarray(
            (512,
            512),
            buffer=np.array(bytes(dict_frame[device_id][-1])),
            dtype=np.uint8)
        resized = cv2.resize(arr, (512 ,512))
        data1 = resized
        im = Image.fromarray(data1)
        im.save("output/"+str(device_id)+"/"+str(count)+".jpeg")
        await asyncio.sleep(1)
        Process(target= await detect(source="output/"+str(device_id)+"/"+str(count)+".jpeg", device_id=device_id)).start()
        print("image saved")
        count += 1
    except TypeError as e:
            print(TypeError,"error" ,e)
                
async def stream_thread(device_id , frame_byte,timestamp, geo_location) :
    if len(dict_frame) == 0 :
        dict_frame[device_id] = list(frame_byte)
        count_frame[device_id] = 1 
    else:
        for item in list(dict_frame.keys()):
            if item == device_id: 
                dict_frame[device_id].append(list(frame_byte))
                count_frame[device_id] += 1
            else:
                dict_frame[device_id] = list(frame_byte)
                count_frame[device_id] = 1
    print(count_frame, "count frame", threading.get_ident(),"Threading Id" ,device_id ,"Device id")
    await asyncio.sleep(1)
    Process(target = await save_image(device_id=device_id)).start()


async def cb1(msg):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try :
        data =(msg.data)
        data = data.decode('ISO-8859-1')
        parse = json.loads(data)
        device_id = parse['device_id']
        frame_code = parse['frame_bytes']
        timestamp = parse['timestamp']
        geo_location = parse['geo-location']
        frame_byte = frame_code.encode('ISO-8859-1')
        await asyncio.sleep(1)
        Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location)).start()
        print("starting thread", device_id , "Device id", threading.active_count(),"active threading count", threading.get_ident(), "threading id ")
        device_list.append(device_id)
        logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
        logging.debug("Debug logging test...")
        logging.info("Program is working as expected")
        logging.warning("Warning, the program may not function properly")
        logging.error("The program encountered an error")
        logging.critical("The program crashed")
    finally:
        print("done with work ")
        sem.release()

async def cb2(msg):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try :
        data =(msg.data)
        data = data.decode('ISO-8859-1')
        parse = json.loads(data)
        device_id = parse['device_id']
        frame_code = parse['frame_bytes']
        timestamp = parse['timestamp']
        geo_location = parse['geo-location']
        frame_byte = frame_code.encode('ISO-8859-1')
        await asyncio.sleep(1)
        Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location)).start()
        print("starting thread", device_id , "Device id", threading.active_count(),"active threading count", threading.get_ident(), "threading id ")
        device_list.append(device_id)
        logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
        logging.debug("Debug logging test...")
        logging.info("Program is working as expected")
        logging.warning("Warning, the program may not function properly")
        logging.error("The program encountered an error")
        logging.critical("The program crashed")
    finally:
        print("done with work ")
        sem.release()
        
async def cb3(msg):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try :
        data =(msg.data)
        data = data.decode('ISO-8859-1')
        parse = json.loads(data)
        device_id = parse['device_id']
        frame_code = parse['frame_bytes']
        timestamp = parse['timestamp']
        geo_location = parse['geo-location']
        frame_byte = frame_code.encode('ISO-8859-1')
        await asyncio.sleep(1)
        Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location)).start()
        print("starting thread", device_id , "Device id", threading.active_count(),"active threading count", threading.get_ident(), "threading id ")
        device_list.append(device_id)
        logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
        logging.debug("Debug logging test...")
        logging.info("Program is working as expected")
        logging.warning("Warning, the program may not function properly")
        logging.error("The program encountered an error")
        logging.critical("The program crashed")
    finally:
        print("done with work ")
        sem.release()

async def cb4(msg):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try :
        data =(msg.data)
        data = data.decode('ISO-8859-1')
        parse = json.loads(data)
        device_id = parse['device_id']
        frame_code = parse['frame_bytes']
        timestamp = parse['timestamp']
        geo_location = parse['geo-location']
        frame_byte = frame_code.encode('ISO-8859-1')
        await asyncio.sleep(1)
        Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location)).start()
        print("starting thread", device_id , "Device id", threading.active_count(),"active threading count", threading.get_ident(), "threading id ")
        device_list.append(device_id)
        logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
        logging.debug("Debug logging test...")
        logging.info("Program is working as expected")
        logging.warning("Warning, the program may not function properly")
        logging.error("The program encountered an error")
        logging.critical("The program crashed")
    finally:
        print("done with work ")
        sem.release()
        
async def cb5(msg):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try :
        data =(msg.data)
        data = data.decode('ISO-8859-1')
        parse = json.loads(data)
        device_id = parse['device_id']
        frame_code = parse['frame_bytes']
        timestamp = parse['timestamp']
        geo_location = parse['geo-location']
        frame_byte = frame_code.encode('ISO-8859-1')
        await asyncio.sleep(1)
        Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location)).start()
        print("starting thread", device_id , "Device id", threading.active_count(),"active threading count", threading.get_ident(), "threading id ")
        device_list.append(device_id)
        logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
        logging.debug("Debug logging test...")
        logging.info("Program is working as expected")
        logging.warning("Warning, the program may not function properly")
        logging.error("The program encountered an error")
        logging.critical("The program crashed")
    finally:
        print("done with work ")
        sem.release()
        
async def cb6(msg):
    global count 
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    try :
        data =(msg.data)
        data = data.decode('ISO-8859-1')
        parse = json.loads(data)
        device_id = parse['device_id']
        frame_code = parse['frame_bytes']
        timestamp = parse['timestamp']
        geo_location = parse['geo-location']
        frame_byte = frame_code.encode('ISO-8859-1')
        await asyncio.sleep(1)
        t = Process(target= await stream_thread(device_id=device_id ,frame_byte=frame_byte, timestamp=timestamp, geo_location=geo_location))
        
        if device_id not in set(device_list) or len(device_list) < 2:
            t.start()
            print("starting thread", device_id , "Device id", threading.active_count(),"active threading count", threading.get_ident(), "threading id ")
        device_list.append(device_id)
        logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
        logging.debug("Debug logging test...")
        logging.info("Program is working as expected")
        logging.warning("Warning, the program may not function properly")
        logging.error("The program encountered an error")
        logging.critical("The program crashed")
    finally:
        print("done with work ")
        sem.release()
        

async def main():
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 20 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    Process(target = await js.subscribe("stream.1.frame", cb=cb1, stream="device_stream" , idle_heartbeat = 2)).start()
    # await js.subscribe("stream.1.frame", cb=cb1, stream="device_stream" , idle_heartbeat = 2)
    Process(target = await js.subscribe("stream.2.frame", cb=cb2 ,stream="device_stream" ,idle_heartbeat = 2)).start()
    # await js.subscribe("stream.2.frame", cb=cb2, stream="device_stream" , idle_heartbeat = 2)
    # await js.subscribe("stream.3.frame", cb=cb3, stream="device_stream" , idle_heartbeat = 2)
    # await js.subscribe("stream.3.frame", cb=cb3, stream="device_stream" , idle_heartbeat = 2)
    # await js.subscribe("stream.4.frame", cb=cb4 ,stream="device_stream" ,idle_heartbeat = 2)
    # await js.subscribe("stream.4.frame", cb=cb4, stream="device_stream" , idle_heartbeat = 2)
    # await js.subscribe("stream.5.frame", cb=cb5, stream="device_stream" , idle_heartbeat = 2)
    # await js.subscribe("stream.5.frame", cb=cb5, stream="device_stream" , idle_heartbeat = 2)
    # await js.subscribe("stream.6.frame", cb=cb6,stream="device_stream" ,idle_heartbeat = 2)
    # await js.subscribe("stream.6.frame", cb=cb6,stream="device_stream" ,idle_heartbeat = 2)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")