import asyncio
from itertools import count
from queue import Empty
import nats
import json
import numpy as np 
from PIL import Image
import cv2
import torch


Dict_frame = {}
frame = []
count = 0

async def  save_image(device_id):
    global count 
    try :  
        arr = np.ndarray(
            (720,
            720),
            buffer=np.array(bytes(Dict_frame[device_id][-1])),
            dtype=np.uint8)
        resized = cv2.resize(arr, (720 ,720))
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
    #for a single stream 
    # frame.append(frame_byte)
    # Dict_frame[device_id] = frame
    # print(Dict_frame.keys())
    # print(len(Dict_frame[device_id]))
    # if len(Dict_frame[device_id]) % 5 == 0:
    #     # print(Dict_frame, "Dict_frame")
    #     count = 0
    #     print("devie == 5")
    #     for value in Dict_frame[device_id]:   
    #         try :  
    #             arr = np.ndarray(
    #                 (720,
    #                 720),
    #                 buffer=np.array(value),
    #                 dtype=np.uint8)
    #             print(type(value),"Value")
    #             resized = cv2.resize(arr, (720 ,720))
    #             data1 = resized
    #             print("DEVICE_ID :", device_id)
    #             print("TIMESTAMP :", timestamp)
    #             print("Geo-location",geo_location)
    #             im = Image.fromarray(data1)
    #             im.save("output/output"+str(count)+".jpeg")
    #             print("image saved")
    #             count += 1
    #         except TypeError as e:
    #             print(TypeError,"error" ,e)
    #     frame.clear()

    # For Multi Devices 
    if len(Dict_frame) == 0 :
        print("line 56")
        Dict_frame[device_id] = list(frame_byte)
        # await save_image(device_id=device_id)
    else:
        for item in Dict_frame.keys():
            if item == device_id: 
                Dict_frame[device_id].append(list(frame_byte))
                print(len(Dict_frame[device_id]),"Key",item , "Length of Dictonary")
                await save_image(device_id=device_id)
            else:
                Dict_frame[device_id] = list(frame_byte)
                # await save_image(device_id=device_id)
             
    print(Dict_frame.keys())
        

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