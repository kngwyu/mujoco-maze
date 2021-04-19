import asyncio
import io
import multiprocessing as mp

import fastapi
import uvicorn

from PIL import Image

HTML = """
<!DOCTYPE html>
<html>
  <head>
    <title>MuJoCo maze visualizer</title>
  </head>
  <body>
    <script>
      var web_socket = new WebSocket('ws://127.0.0.1:{{port}}/ws');
      web_socket.binaryType = "arraybuffer";
      web_socket.onmessage = function(event) {
          var canvas = document.getElementById('canvas');
          var ctx = canvas.getContext('2d');
          var blob = new Blob([event.data], {type:'image/png'});
          var url = URL.createObjectURL(blob);
          var image = new Image();
          image.onload = function() {
              ctx.drawImage(image, 0, 0);
          }
          console.log(url);
          image.src = url;
      }
    </script>
    <div>
      <canvas id="canvas" width="600" height="480"></canvas>
    </div>
  </body>
</html>
"""


class _ServerWorker(mp.Process):
    def __init__(self, pipe: mp.connection.Pipe, port: int) -> None:
        super().__init__()
        self.pipe = pipe
        self.port = port

    def _run_server(self) -> None:

        app = fastapi.FastAPI()
        html = HTML.replace("{{port}}", str(self.port))

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
