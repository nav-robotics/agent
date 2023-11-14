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
import numpy as np
from pydantic import BaseModel
import v4l2py
import cv2
import json


def parse_video_capture_devices():
    video_capture_devices = list(v4l2py.iter_video_capture_devices())

    parsed_video_capture_devices = []

    for device in video_capture_devices:
        with device:
            parsed_video_capture_devices.append(device)

    return parsed_video_capture_devices


black_frame = np.zeros((640, 480, 3), dtype=np.uint8)


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


class CameraStreamTrack(MediaStreamTrack):
    kind = "video"

    _started = False
    _start: float
    _timestamp: int

    video_captures = {}

    frame_rate = 1 / 30
    clock_rate = 90000
    time_base = fractions.Fraction(1, clock_rate)

    def __init__(self, index=0, apiPreference=cv2.CAP_V4L2, params=[]):
        super().__init__()
        self.index = index

        if index not in self.video_captures:
            self.video_captures[index] = cv2.VideoCapture(
                index=index,
                apiPreference=apiPreference,
                params=params,
            )

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
        success, frame = self.video_captures[self.index].read()
        return VideoFrame.from_ndarray(
            array=frame if success else black_frame, format="bgr24"
        )


def main():
    peer_connections = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        coroutines = [connection.close() for connection in peer_connections]
        await asyncio.gather(*coroutines)

        for video_capture in CameraStreamTrack.video_captures.values():
            video_capture.release()

        cv2.destroyAllWindows()
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
        video_capture_devices = parse_video_capture_devices()

        devices = [
            {
                "filename": device.filename,
                "index": device.index,
                "driver": device.info.driver,
                "card": device.info.card,
                "bus_info": device.info.bus_info,
                "version": device.info.version,
                "capabilities": device.info.capabilities,
                "device_capabilities": device.info.device_capabilities,
                "crop_capabilities": device.info.crop_capabilities,
                "buffers": device.info.buffers,
                "formats": device.info.formats,
                "inputs": device.info.inputs,
                "controls": device.info.controls,
                "frame_sizes": [
                    {
                        "pixel_format": frame_size.pixel_format.human_str(),
                        "width": frame_size.width,
                        "height": frame_size.height,
                        "max_fps": str(frame_size.max_fps),
                        "min_fps": str(frame_size.min_fps),
                        "step_fps": str(frame_size.step_fps),
                    }
                    for frame_size in device.info.frame_sizes
                ],
            }
            for device in video_capture_devices
        ]

        return devices

    @app.post("/offer")
    async def post_offer(offer: WebRTCOffer):
        peer_connection = RTCPeerConnection()
        peer_connections.add(peer_connection)

        @peer_connection.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            async def on_message(message):
                match channel.label:
                    case "renegotiation":
                        print("renegotiation")

                        offer = json.loads(message)

                        description = RTCSessionDescription(
                            sdp=offer["sdp"], type=offer["type"]
                        )
                        await peer_connection.setRemoteDescription(description)

                        answer = await peer_connection.createAnswer()
                        await peer_connection.setLocalDescription(answer)

                        channel.send(
                            json.dumps(
                                {
                                    "sdp": peer_connection.localDescription.sdp,
                                    "type": peer_connection.localDescription.type,
                                }
                            )
                        )

                    case "status":
                        match message:
                            case "disconnected":
                                await peer_connection.close()
                                peer_connections.discard(peer_connection)

                    case "track":
                        options = json.loads(message)
                        track = CameraStreamTrack(options["index"])
                        peer_connection.addTrack(track)
                        channel.send("")

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            print("connection: ", peer_connection.connectionState)

            match peer_connection.connectionState:
                case "failed":
                    await peer_connection.close()
                    peer_connections.discard(peer_connection)

        description = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        await peer_connection.setRemoteDescription(description)

        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)

        return peer_connection.localDescription

    return app


if __name__ == "__main__":
    main()
