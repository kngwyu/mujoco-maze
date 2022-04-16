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
    <h2>MuJoCo Maze Visualizer</h2>
    <script>
      var ws_image = new WebSocket('ws://127.0.0.1:{{port}}/ws');
      ws_image.binaryType = "arraybuffer";
      ws_image.onmessage = function(event) {
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
      <canvas id="canvas" width="1200" height="800"></canvas>
    </div>
    <a href="video">Video</a>
  </body>
</html>
"""


class _ServerWorker(mp.Process):
    def __init__(self, pipe: mp.connection.Pipe, port: int) -> None:
        super().__init__()
        self.pipe = pipe
        self.port = port
        self.video_frames = []

    def _run_server(self) -> None:

        app = fastapi.FastAPI()
        html = HTML.replace("{{port}}", str(self.port))

        @app.get("/")
        async def root():
            return fastapi.responses.HTMLResponse(html)

        server = None

        @app.websocket("/ws")
        async def ws(websocket: fastapi.WebSocket):
            await websocket.accept()
            loop = asyncio.get_running_loop()
            while True:
                image_array = await loop.run_in_executor(None, self.pipe.recv)
                if image_array is None:
                    break
                self.video_frames.append(image_array)
                image = Image.fromarray(image_array)
                with io.BytesIO() as stream:
                    image.save(stream, format="png")
                    res = stream.getvalue()
                    await websocket.send_bytes(res)
            await websocket.close()
            server.should_exit = True

        @app.get("/video")
        async def video():
            import imageio

            writer = imageio.get_writer("/tmp/mujoco-maze-video.mp4")
            for frame in self.video_frames:
                writer.append_data(frame)
            writer.close()
            video = open("/tmp/mujoco-maze-video.mp4", mode="rb")
            return fastapi.responses.StreamingResponse(video, media_type="video/mp4")

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
