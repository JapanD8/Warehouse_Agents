from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import json
import asyncio
from agents import agent_graph

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (Frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        csv_text = content.decode("utf-8")
        
        # Initial State
        initial_state = {
            "messages": [],
            "csv_content": csv_text,
            "placed_items": [],
            "errors": []
        }
        
        # Run the graph and collect outputs
        # We need to collect final state to send back, 
        # OR we could stream if we use WebSocket. 
        # For simplicity in this endpoint, we return the final result.
        
        final_state = await agent_graph.ainvoke(initial_state)
        
        result = {
            "messages": final_state.get("messages", []),
            "items": final_state.get("placed_items", []),
            "floor": final_state.get("floor_dims", {}),
            "errors": final_state.get("errors", [])
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

# WebSocket for "Live" updates if desired
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting CSV content in data for simplicity or a JSON command
            message = json.loads(data)
            
            if message.get("type") == "run_plan":
                csv_text = message.get("csv")
                initial_state = {
                    "messages": ["Received plan request."],
                    "csv_content": csv_text,
                    "placed_items": [],
                    "errors": []
                }
                
                # Stream the graph execution
                async for event in agent_graph.astream(initial_state):
                    # event is a dict of node_name: state_update
                    for node, state_update in event.items():
                        # We extract relevant info to send to frontend
                        response = {
                            "type": "update",
                            "node": node,
                            "messages": state_update.get("messages", []),
                            "items": state_update.get("placed_items", []),
                            # Note: placed_items in the update might be just the delta if we defined reducer,
                            # or the full list depending on Annotated behavior. 
                            # In our agents.py, 'placed_items' is Annotated with operator.add, 
                            # so the 'state_update' from the node typically contains just the NEW items added by that node 
                            # (because standard LangGraph node functions return the *update* to the state).
                            # Let's verify: In agents.py, rack_agent returns {"placed_items": placed...}.
                            # Since it's operator.add, these will be APPENDED to global state.
                            # So sending them to frontend is "new items".
                        }
                        await websocket.send_json(response)
                        await asyncio.sleep(0.1) # Debounce slightly for visual effect
                
                await websocket.send_json({"type": "complete"})

    except Exception as e:
        print(f"WS Error: {e}")

@app.get("/download", response_class=HTMLResponse)
async def download_scene():
    try:
        # 1. Read Memory
        if not os.path.exists("warehouse_memory.json"):
            return HTMLResponse("<h1>Error: No warehouse data found. Run planner first.</h1>")
            
        with open("warehouse_memory.json", "r") as f:
            memory_data = f.read()
            
        # 2. Read JS Logic
        js_path = os.path.join("static", "js", "three_scene.js")
        if not os.path.exists(js_path):
             return HTMLResponse("<h1>Error: JS logic not found.</h1>")
             
        with open(js_path, "r") as f:
            js_content = f.read()
            
        # 3. Read Template
        if not os.path.exists("export_template.html"):
            return HTMLResponse("<h1>Error: Template not found.</h1>")
            
        with open("export_template.html", "r") as f:
            template = f.read()
            
        # 4. Inject
        # We need to make sure JS doesn't have syntax errors when injected
        # The template expects __SCENE_DATA__ and __THREE_JS_LOGIC__
        
        final_html = template.replace("__SCENE_DATA__", memory_data)
        final_html = final_html.replace("__THREE_JS_LOGIC__", js_content)
        
        # Force download header? Or just view?
        # User asked for "save", so let's make it downloadable or just openable.
        # "Content-Disposition: attachment" forces download
        
        return HTMLResponse(content=final_html, headers={"Content-Disposition": "attachment; filename=warehouse_scene.html"})

    except Exception as e:
        return HTMLResponse(f"<h1>Error generating export: {str(e)}</h1>")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
