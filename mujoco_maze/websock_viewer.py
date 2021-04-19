import asyncio
import io
import multiprocessing as mp
import pathlib

import fastapi
import uvicorn

from PIL import Image


class _ServerWorker(mp.Process):
    def __init__(self, pipe: mp.connection.Pipe, port: int) -> None:
        super().__init__()
        self.pipe = pipe
        self.port = port

    def _run_server(self) -> None:

        app = fastapi.FastAPI()
        static = pathlib.Path(__file__).parent.joinpath("static")
        html_path = static.joinpath("index.html")
        html = html_path.read_text().replace("{{port}}", str(self.port))

        @app.get("/")
        async def get():
            return fastapi.responses.HTMLResponse(html)

        server = None

        @app.websocket("/ws")
        async def ws_send_image(websocket: fastapi.WebSocket):
            await websocket.accept()
            loop = asyncio.get_running_loop()
            while True:
                image_array = await loop.run_in_executor(None, self.pipe.recv)
                if image_array is None:
                    break
                image = Image.fromarray(image_array)
                with io.BytesIO() as stream:
                    image.save(stream, format="png")
                    res = stream.getvalue()
                    await websocket.send_bytes(res)
            await websocket.close()
            server.should_exit = True

        config = uvicorn.Config(app, port=self.port)
        server = uvicorn.Server(config)
        server.run()

    def run(self) -> None:
        try:
            self._run_server()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("Exception in websocket server")
            raise e


def start_server(port: int) -> mp.connection.Connection:
    mainproc_pipe, server_pipe = mp.Pipe()
    worker = _ServerWorker(server_pipe, port)
    worker.start()
    return mainproc_pipe
