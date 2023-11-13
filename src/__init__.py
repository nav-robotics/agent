from fastapi import FastAPI
from contextlib import asynccontextmanager
from aiortc.contrib.media import MediaStreamTrack
from aiortc import RTCPeerConnection, RTCSessionDescription
from typing import Tuple
import fractions
from av.frame import Frame
from av import VideoFrame
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import numpy as np
from pydantic import BaseModel
import v4l2py
import cv2

black_frame = np.zeros((640, 480, 3), dtype=np.uint8)


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


def get_devices_info():
    devices = list(v4l2py.iter_video_capture_devices())

    devices_info = []

    for device in devices:
        with device:
            print(device.get_format(v4l2py.device.BufferType.VIDEO_CAPTURE))
            devices_info.append(device.info)

    return devices_info


class CameraStreamTrack(MediaStreamTrack):
    kind = "video"

    _started = False
    _start: float
    _timestamp: int

    frame_rate = 1 / 30
    clock_rate = 90000
    time_base = fractions.Fraction(1, clock_rate)

    def __init__(self):
        super().__init__()
        self.video_capture = cv2.VideoCapture(0)
        self.device = v4l2py.Device.from_id(2)
        self.device.open()
        self.stream = iter(self.device)

    async def recv(self) -> Frame:
        await self.next_timestamp()

        frame = await self.get_frame()
        frame.pts = self._timestamp
        frame.time_base = self.time_base

        return frame

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self._started:
            self._timestamp += int(self.frame_rate * self.clock_rate)
            await asyncio.sleep(
                self._start + (self._timestamp / self.clock_rate) - time.time()
            )
        else:
            self._start = time.time()
            self._timestamp = 0
            self._started = True

    async def get_frame(self):
        frame = next(self.stream)
        data = frame.array
        data.shape = frame.height, frame.width, -1
        bgr = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_YUYV)
        return VideoFrame.from_ndarray(array=bgr, format="bgr24")


def main():
    peer_connections = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        coroutines = [peer_connection.close() for peer_connection in peer_connections]
        await asyncio.gather(*coroutines)
        peer_connections.clear()

    app = FastAPI(lifespan=lifespan)

    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def get_index():
        return {"message": "Hello World"}

    @app.get("/devices")
    def get_devices():
        devices_info = get_devices_info()
        devices = [
            {
                "driver": device_info.driver,
                "card": device_info.card,
                "version": device_info.version,
                "capabilities": device_info.capabilities,
                "device_capabilities": device_info.device_capabilities,
                "buffers": device_info.buffers,
            }
            for device_info in devices_info
        ]

        return devices

    @app.post("/offer")
    async def post_offer(offer: WebRTCOffer):
        peer_connection = RTCPeerConnection()
        peer_connections.add(peer_connection)

        @peer_connection.on("datachannel")
        def on_datachannel(channel):
            print(channel.label, "-", "created by remote party")

            @channel.on("message")
            async def on_message(message):
                match channel.label:
                    case "status":
                        if message == "disconnected":
                            await peer_connection.close()
                            peer_connections.discard(peer_connection)
                        else:
                            print(channel.label, " received message: ", message)

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            print("connection: ", peer_connection.connectionState, peer_connection)

            match peer_connection.connectionState:
                case "failed":
                    await peer_connection.close()
                    peer_connections.discard(peer_connection)

        track = CameraStreamTrack()
        peer_connection.addTrack(track)

        description = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        await peer_connection.setRemoteDescription(description)

        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)

        return peer_connection.localDescription

    run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
