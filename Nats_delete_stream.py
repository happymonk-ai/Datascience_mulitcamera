import asyncio
from asyncio import streams
import nats
from nats.errors import TimeoutError 
from nats.js.errors import NotFoundError


import nest_asyncio
nest_asyncio.apply()

async def error_cb(e):
    print("There was an Error:{e}")


async def main(delay):
    await asyncio.sleep(delay)
    # nc = await nats.connect(servers=["nats://216.48.189.5:4222"])
    # nc = await nats.connect(servers=["nats://216.48.181.154:4222"] , error_cb =error_cb ,reconnect_time_wait=2 ,allow_reconnect=True, pedantic = True)
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , error_cb =error_cb , reconnect_time_wait= 5 ,allow_reconnect=True)
    js = nc.jetstream()
    
    while True:
        try:
            # await js.delete_stream("Testing_stream1")
            await js.delete_stream("device_stream")
            print("Task Completed","\nStream deleted")
            break
        except TimeoutError:
            print("Error ***")
            print("Server not responding")
            break
        except NotFoundError:
            print("Error ***")
            print("Stream Not Found")
            break
            

async def run():
    task1 = asyncio.create_task(main(0))
    await task1


if __name__ == '__main__':
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    # loop.run_forever()
    # loop.close()
