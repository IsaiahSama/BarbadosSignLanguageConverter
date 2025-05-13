from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse

from bslapi import ImageNNApi


app = FastAPI()


@app.get("/")
async def index(request: Request):
    return {"message": "Hello World"}


@app.post("/recognize/image")
async def recognize_image(request: Request):
    pass


@app.post("/recognize/video")
async def recognize_video(request: Request):
    pass


@app.websocket("/ws/recognize/image")
async def ws_recognize_image(websocket: WebSocket):
    pass


@app.websocket("/ws/recognize/video")
async def ws_recognize_video(websocket: WebSocket):
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", port=5500)
