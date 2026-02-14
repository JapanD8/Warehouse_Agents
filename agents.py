import pandas as pd
import io
import math
import logging
import os
import json
from typing import TypedDict, List, Dict, Any, Union, Annotated
from langgraph.graph import StateGraph, END
import operator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.utils.json import parse_json_markdown

# Load Env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("geminiapikey")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True
)

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]  # Chat log (append-only)
    csv_content: str
    floor_dims: Dict[str, float]  # {width, length, height}
    racks_to_place: List[Dict[str, Any]]  # List of rack specs
    placed_items: Annotated[List[Dict[str, Any]], operator.add] 
    errors: List[str]
    next_step: str

# --- Agents ---

# 1. Planner Agent: Pure LLM (No Fallback)
def planner_agent(state: AgentState):
    print("--- Planner Agent (LLM) ---")
    new_messages = ["Planner: Received input, analyzing layout requirements..."]
    
    content_text = state.get('csv_content', "")
    
    if not content_text:
        return {"messages": ["Planner: Error - No content provided for analysis."], "errors": ["No content"]}

    floor_dims = {}
    racks_to_place = []
    
    # Attempt LLM Parsing
    try:
        print("Sending prompt to Gemini...")
        
        system_prompt = """You are a Warehouse Planning Agent.
        Your job is to extract the floor dimensions and a list of racks to be placed from the user's input.
        The input might be a CSV file content or a natural language description.
        
        Return ONLY valid JSON with this structure:
        {
            "floor": {"width": float, "length": float, "height": float},
            "racks": [
                {"count": int, "width": float, "length": float, "height": float}
            ]
        }
        If dimensions are missing, make reasonable assumptions (Floor: 60x40x12, Rack: 2.5x1.2x6).
        """
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Input Data:\n{content_text}")
        ])
        
        # Log the raw response so the user sees it's AI
        raw_output_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
        new_messages.append(f"Gemini Response: {raw_output_preview}")
        
        start_plan = parse_json_markdown(response.content)
        floor_dims = start_plan.get('floor', {'width': 60, 'length': 40, 'height': 12})
        racks_data = start_plan.get('racks', [])
        
        rack_id_counter = 1
        for r_batch in racks_data:
            count = r_batch.get('count', 1)
            for _ in range(count):
                racks_to_place.append({
                    'id': f"rack_{rack_id_counter}",
                    'w': float(r_batch.get('width', 2.5)),
                    'l': float(r_batch.get('length', 1.2)),
                    'h': float(r_batch.get('height', 6.0))
                })
                rack_id_counter += 1
                
        new_messages.append("Planner: Layout requirements extracted. Proceeding to floor setup.")

    except Exception as e:
        print(f"LLM Failed: {e}")
        return {"messages": [f"Planner: Error extracting layout requirements ({str(e)})."], "errors": [str(e)]}

    return {
        "floor_dims": floor_dims, 
        "racks_to_place": racks_to_place, 
        "messages": new_messages,
        "next_step": "floor_agent"
    }

# 2. Floor Agent: Prepares floor geometry data
def floor_agent(state: AgentState):
    print("--- Floor Agent ---")
    new_messages = ["Floor Agent: Building floor and wall geometry..."]
    
    dims = state['floor_dims']
    # Create the floor item
    floor_item = {
        "type": "floor",
        "width": dims['width'],
        "length": dims['length'],
        "height": dims['height']
    }
    # Create wall item
    walls_item = {
        "type": "walls",
        "width": dims['width'],
        "length": dims['length'],
        "height": dims['height']
    }
    
    new_messages.append("Floor Agent: Floor built successfully.")
    new_messages.append("Planner: Assigning Rack Agent to build racks...")
    
    return {
        "placed_items": [floor_item, walls_item],
        "messages": new_messages,
        "next_step": "rack_agent"
    }

# 3. Rack Agent: Places racks with simple collision detection & Saves Memory
def rack_agent(state: AgentState):
    print("--- Rack Agent ---")
    new_messages = ["Rack Agent: Received task from Planner."]
    
    floor = state['floor_dims']
    racks = state['racks_to_place']
    placed = []
    
    # Floor boundaries
    half_w = floor['width'] / 2
    half_l = floor['length'] / 2
    
    # Margins and Aisles
    margin = 2.0
    aisle_gap = 2.5 
    
    current_x = -half_w + margin
    current_z = -half_l + margin
    
    successful_placements = 0
    total_rack_area = 0
    
    for rack in racks:
        w = rack['w']
        l = rack['l']
        h = rack['h']
        
        # Check if fits in current row
        if current_x + w > half_w - margin:
            current_x = -half_w + margin
            current_z += 6.0 + aisle_gap # approximate row depth + aisle
        
        # Check if fits in floor (Z)
        if current_z + l > half_l - margin:
            break
            
        # Place it
        pos_x = current_x + w/2
        pos_z = current_z + l/2
        
        item = {
            "type": "rack",
            "id": rack['id'],
            "x": pos_x,
            "z": pos_z,
            "w": w,
            "l": l,
            "h": h,
            "occupied_area": w * l
        }
        placed.append(item)
        successful_placements += 1
        total_rack_area += (w * l)
        
        current_x += w + 0.5
        
    # --- MEMORY PERSISTENCE ---
    floor_area = floor['width'] * floor['length']
    free_area = floor_area - total_rack_area
    
    memory_snapshot = {
        "floor": floor,
        "stats": {
            "total_area": floor_area,
            "occupied_area": total_rack_area,
            "free_area": free_area,
            "utilization": f"{(total_rack_area/floor_area)*100:.2f}%"
        },
        "occupied_spaces": [
            {"id": p['id'], "x": p['x'], "z": p['z'], "w": p['w'], "l": p['l']} 
            for p in placed
        ]
    }
    
    # Save to file
    with open("warehouse_memory.json", "w") as f:
        json.dump(memory_snapshot, f, indent=2)
    
    new_messages.append(f"Rack Agent: Racks updated. ({successful_placements} placed)")
    new_messages.append("Planner: Workflow complete.")
    
    return {
        "placed_items": placed,
        "messages": new_messages,
        "next_step": "end"
    }

# --- Graph ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_agent)
    workflow.add_node("floor", floor_agent)
    workflow.add_node("racks", rack_agent)
    
    workflow.set_entry_point("planner")
    
    def router(state):
        if state.get("errors"):
            return END
        return state['next_step']
        
    workflow.add_conditional_edges(
        "planner",
        router,
        {
            "floor_agent": "floor",
            END: END
        }
    )
    
    workflow.add_edge("floor", "racks")
    workflow.add_edge("racks", END)
    
    return workflow.compile()

agent_graph = build_graph()
