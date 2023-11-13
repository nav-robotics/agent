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
import cv2
import logging
from pydantic import BaseModel


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


aiortc_logger = logging.getLogger("aiortc")
aiortc_logger.setLevel(logging.DEBUG)

counter = 0

cameras = [0, 4, 8, 10]

frame_rate = 1 / 30

clock_rate = 90000

time_base = fractions.Fraction(1, clock_rate)

black_frame = np.zeros((640, 480, 3), dtype=np.uint8)


class RosVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    _started = False
    _start: float
    _timestamp: int

    def __init__(self):
        super().__init__()

        global counter
        self.counter = counter

        # self.video_capture = nanocamera.Camera(camera_type=1, device_id=counter, width=640, height=480, fps=30)

        self.video_capture = cv2.VideoCapture(cameras[counter])

        # cv2.CAP_V4L, params=[
        #     cv2.CAP_PROP_FRAME_WIDTH, 640,
        #     cv2.CAP_PROP_FRAME_HEIGHT, 480,
        #     cv2.CAP_PROP_FPS, 20
        # ])

        counter = counter + 1 if counter < 3 else 0

    async def recv(self) -> Frame:
        await self.next_timestamp()

        frame = await self.get_frame()
        frame.pts = self._timestamp
        frame.time_base = time_base

        # print(self.counter, frame.pts, frame.time_base)

        return frame

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self._started:
            self._timestamp += int(frame_rate * clock_rate)
            await asyncio.sleep(
                self._start + (self._timestamp / clock_rate) - time.time()
            )
        else:
            self._start = time.time()
            self._timestamp = 0
            self._started = True

    async def get_frame(self):
        success, frame = self.video_capture.read()
        return VideoFrame.from_ndarray(
            array=frame if success else black_frame, format="bgr24"
        )


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

    @app.post("/offer")
    async def post_offer(offer: WebRTCOffer):
        peer_connection = RTCPeerConnection()
        peer_connections.add(peer_connection)

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            print("connection: ", peer_connection.connectionState, peer_connection)

            match peer_connection.connectionState:
                case "failed":
                    await peer_connection.close()
                    peer_connections.discard(peer_connection)

        track = RosVideoStreamTrack()
        peer_connection.addTrack(track)

        description = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        await peer_connection.setRemoteDescription(description)

        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)

        return peer_connection.localDescription

    run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
