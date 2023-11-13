from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from aiortc.contrib.media import MediaStreamTrack
from aiortc import RTCPeerConnection, RTCSessionDescription
from typing import Tuple
import cv2
import fractions
from av.frame import Frame
from av import VideoFrame
import time
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run


class RosVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    _started = False
    _start: float
    _timestamp: int

    clock_rate = 90000
    frame_rate = 1 / 30
    time_base = fractions.Fraction(1, clock_rate)

    def __init__(self):
        super().__init__()
        self.video_capture = cv2.VideoCapture("video.mp4")

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
        _, array = self.video_capture.read()
        return VideoFrame.from_ndarray(array=array, format="bgr24")


def main():
    peer_connections = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        coroutines = [connection.close() for connection in peer_connections]
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
    def index():
        return {"message": "Hello World"}

    @app.post("/offer")
    async def offer(request: Request):
        peer_connection = RTCPeerConnection()

        peer_connections.add(peer_connection)

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange():
            match peer_connection.connectionState:
                case "failed":
                    await peer_connection.close()
                    peer_connections.discard(peer_connection)

        peer_connection.addTrack(RosVideoStreamTrack())

        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await peer_connection.setRemoteDescription(offer)

        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)

        return peer_connection.localDescription

    run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
