import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes.chat import router as chat_router, get_agent
from routes.message_assistant import router as message_assistant_router
from routes.expertise import router as expertise_router
from routes.matching import router as matching_router

import time
import asyncio
import httpx
from fastapi.responses import HTMLResponse

# 0. Global State for Dashboard
start_time = time.time()
heartbeats = []

async def self_pinger():
    """Background task to keep the service awake by pinging itself."""
    await asyncio.sleep(10) # Wait for server to be fully ready
    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Use localhost for internal pings
                port = int(os.getenv("PORT", 8000))
                await client.get(f"http://localhost:{port}/")
                heartbeats.insert(0, time.strftime("%H:%M:%S"))
                if len(heartbeats) > 10: heartbeats.pop()
                print(f"🚀 [HEARTBEAT] AI Agent self-pinged at {heartbeats[0]}")
        except Exception as e:
            print(f"⚠️ [HEARTBEAT] Self-ping failed: {e}")
        
        await asyncio.sleep(12 * 60) # 12 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the CliQ Agent on startup
    print("🚀 Initializing CliQ AI Agent (OOP Mode)...")
    get_agent()
    print("✅ CliQ AI Agent ready.")
    
    # Start self-pinger background task
    asyncio.create_task(self_pinger())
    
    yield

# Configure CORS
frontend_url = os.getenv("FRONTEND_URL", "*")
origins = [frontend_url] if frontend_url != "*" else ["*"]

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(message_assistant_router, prefix="/api/message-ai")
app.include_router(expertise_router, prefix="/api/expertise")
app.include_router(matching_router, prefix="/api/match")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def read_root():
    uptime_seconds = int(time.time() - start_time)
    h = uptime_seconds // 3600
    m = (uptime_seconds % 3600) // 60
    s = uptime_seconds % 60
    uptime_str = f"{h}h {m}m {s}s"
    
    heartbeat_items = "".join([f'<div class="heartbeat-item">SUCCESS <span>{hb}</span></div>' for hb in heartbeats])
    if not heartbeat_items:
        heartbeat_items = '<div style="opacity:0.3">Waiting for first heartbeat...</div>'

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CliQ AI Agent | Status</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0a0a0a; color: #fff; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
            .card {{ background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); border-radius: 24px; padding: 40px; width: 90%; max-width: 450px; text-align: center; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }}
            .glow-icon {{ width: 80px; height: 80px; background: #8800ff; border-radius: 50%; margin: 0 auto 24px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 30px #8800ff; animation: pulse 2s infinite; }}
            @keyframes pulse {{ 0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(136, 0, 255, 0.7); }} 70% {{ transform: scale(1); box-shadow: 0 0 0 20px rgba(136, 0, 255, 0); }} 100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(136, 0, 255, 0); }} }}
            h1 {{ font-size: 24px; margin-bottom: 8px; font-weight: 700; color: #8800ff; }}
            p.status {{ font-size: 14px; opacity: 0.6; margin-bottom: 32px; letter-spacing: 1px; text-transform: uppercase; }}
            .stat-box {{ background: rgba(255,255,255,0.03); border-radius: 16px; padding: 20px; margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.05); }}
            .stat-label {{ font-size: 12px; opacity: 0.5; margin-bottom: 8px; text-transform: uppercase; }}
            .stat-value {{ font-size: 32px; font-weight: 700; font-variant-numeric: tabular-nums; }}
            .heartbeat-list {{ text-align: left; background: rgba(0,0,0,0.2); border-radius: 12px; padding: 16px; font-size: 13px; font-family: monospace; }}
            .heartbeat-title {{ font-size: 11px; opacity: 0.4; margin-bottom: 12px; font-family: sans-serif; text-transform: uppercase; }}
            .heartbeat-item {{ display: flex; justify-content: space-between; margin-bottom: 6px; color: #8800ff; }}
            .heartbeat-item span {{ opacity: 0.5; color: #fff; }}
        </style>
        <script>
            setTimeout(() => location.reload(), 30000); // Auto refresh every 30s
        </script>
    </head>
    <body>
        <div class="card">
            <div class="glow-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <h1>AGENT ACTIVE</h1>
            <p class="status">CliQ AI Professional Service</p>
            
            <div class="stat-box">
                <div class="stat-label">Current Uptime</div>
                <div class="stat-value">{uptime_str}</div>
            </div>

            <div class="heartbeat-list">
                <div class="heartbeat-title">Recent Internal Heartbeats</div>
                {heartbeat_items}
            </div>

            <p style="margin-top: 32px; font-size: 10px; opacity: 0.3;">AUTO-REFRESH EVERY 30S | SELF-PING EVERY 12M</p>
        </div>
    </body>
    </html>
    """
