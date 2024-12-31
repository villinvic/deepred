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
            "color": "#a30000"
        }
        self.ws_address = "wss://transdimensional.xyz/broadcast"

        self.websocket = None
        self.coord_list = []

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(
            self.establish_wc_connection()
        )

        self.stream_step_counter = 0
        self.upload_interval = 500

    def send(self, gamestate: GameState, sprite_id: int = 0):
        self.coord_list.append([gamestate.pos_x, gamestate.pos_y, gamestate.map.value])

        if self.stream_step_counter >= self.upload_interval:
            self.stream_metadata.update(
                extra="",
                sprite_id=sprite_id,
            )
            self.loop.run_until_complete(

                self.broadcast_ws_message(
                    json.dumps(
                        {
                            "metadata": self.stream_metadata,
                            "coords": self.coord_list
                        }
                    )
                )
            )
            self.stream_step_counter = 0
            self.coord_list = []

        self.stream_step_counter += 1


    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException as e:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:
            self.websocket = None