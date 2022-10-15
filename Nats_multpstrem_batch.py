import asyncio
from itertools import count
from queue import Empty
from turtle import shape
import nats
import json
import numpy as np 
from PIL import Image
import cv2
import torch
import os 
from multiprocessing import Process


dict_frame = {}
frame = []
count_frame ={}
count = 0
length = []

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
        print("image saved")
        count += 1
    except TypeError as e:
            print(TypeError,"error" ,e)
    


async def cb(msg):
    global count
    data =(msg.data)
    data = data.decode('ISO-8859-1')
    parse = json.loads(data)
    device_id = parse['device_id']
    frame_code = parse['frame_bytes']
    timestamp = parse['timestamp']
    geo_location = parse['geo-location']
    frame_byte = frame_code.encode('ISO-8859-1')
    # For Multi Devices 
    if len(dict_frame) == 0 :
        dict_frame[device_id] = list(frame_byte)
        count_frame[device_id] = 1 
        await asyncio.sleep(2)
        # await save_image(device_id=device_id)
    else:
        for item in list(dict_frame.keys()):
            if item == device_id: 
                dict_frame[device_id].append(list(frame_byte))
                count_frame[device_id] += 1
                print(count_frame, "batch_list 60")
                await asyncio.sleep(2)
                await save_image(device_id=device_id)
            else:
                dict_frame[device_id] = list(frame_byte)
                count_frame[device_id] = 1
                await asyncio.sleep(2)
                # await save_image(device_id=device_id)
             
    print(dict_frame.keys())
    print(count_frame, "batch_list 61")
        

async def main():
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 5 ,allow_reconnect=True)
    js = nc.jetstream()
      
    await js.subscribe("stream.*.frame", cb=cb, stream="device_stream" , idle_heartbeat = 2)

    await js.subscribe("stream.*.frame", cb=cb ,stream="device_stream" )
    
    # await js.subscribe("test.*.frame", cb=cb, stream="test_stream" , idle_heartbeat = 2)

    # await js.subscribe("test.*.frame", cb=cb ,stream="test_stream")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")