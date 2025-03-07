from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active chat connections
chat_rooms = {}  # {chat_id: [WebSocket1, WebSocket2]}

@app.websocket("/ws/chat/{chat_id}")
async def websocket_chat(websocket: WebSocket, chat_id: str):
    """Allows multiple users (finder & owner) to join the same chat."""
    await websocket.accept()

    # ✅ Add user to chat room
    if chat_id not in chat_rooms:
        chat_rooms[chat_id] = []
    chat_rooms[chat_id].append(websocket)

    try:
        while True:
            message = await websocket.receive_text()

            # ✅ Broadcast message to all users in the chat room
            for conn in chat_rooms[chat_id]:
                if conn != websocket:  # Don't send message back to sender
                    await conn.send_text(f"Other: {message}")

    except WebSocketDisconnect:
        print(f"Client disconnected from chat {chat_id}")
        chat_rooms[chat_id].remove(websocket)
