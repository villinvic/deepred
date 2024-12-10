import asyncio
import time

import websockets
import json
from deepred.polaris_env.gamestate import GameState
from asyncio import Queue

class BotStreamer:
    def __init__(
            self,
            console_id: int = 0,
            bot_name: str = "deepred"
    ):
        self.stream_metadata = {
            "user": bot_name,
            "env_id": console_id,
            "color": "#a30000",
            "extra": "",
        }
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.coord_list = []

        self.queue = Queue()
        self.loop.create_task(self.process_queue())

        self.stream_step_counter = 0
        self.upload_interval = 256

    async def process_queue(self):
        while True:
            message = await self.queue.get()
            await self.broadcast_ws_message(message)
            print(message)

    def send(self, gamestate: GameState):
        self.coord_list.append([gamestate.pos_x, gamestate.pos_y, gamestate.map.value])

        if self.stream_step_counter >= self.upload_interval:
            message = json.dumps(
                {
                    "metadata": self.stream_metadata,
                    "coords": self.coord_list,
                }
            )
            self.queue.put_nowait(message)
            self.stream_step_counter = 0
            self.coord_list = []

        self.stream_step_counter += 1

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:
            self.websocket = None