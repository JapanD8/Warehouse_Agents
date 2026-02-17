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
    forklift_data: Dict[str, Any]
    assigned_area: Dict[str, float]
    assigned_area: Dict[str, float]
    dock_data: Dict[str, Any]
    staging_data: Dict[str, float]
    shipping_data: Dict[str, float]

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
            ],
            "forklifts": {
                "count": int, "width": float, "length": float, "height": float
            },
            "forklift_area": {
                "x": float, "z": float, "width": float, "length": float
            },
            "docks": {
                "side": "top"|"bottom"|"left"|"right", "count": int, "width": float
            },
            "staging": {
                "width": float, "length": float
            },
            "shipping": {
                "width": float, "length": float
            }
        }
        If dimensions are missing, make reasonable assumptions:
        Floor: 60x40x12
        Rack: 2.5x1.2x6
        Forklift: 1.5x2.5x2.5, count=1 if mentioned but no count.
        Forklift Area: Optional. If provided, use it. If not, omit or return null.
        Docks: Optional.
        Staging: Optional. Parse width/length if provided.
        Shipping: Optional. Parse width/length if provided.
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
        racks_data = start_plan.get('racks', [])
        forklift_data = start_plan.get('forklifts', {})
        forklift_area = start_plan.get('forklift_area', None)
        forklift_area = start_plan.get('forklift_area', None)
        dock_data = start_plan.get('docks', {})
        staging_data = start_plan.get('staging', {})
        shipping_data = start_plan.get('shipping', {})
        
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
        "forklift_data": forklift_data,
        "assigned_area": forklift_area, # Initial assignment from LLM if any 
        "assigned_area": forklift_area, # Initial assignment from LLM if any 
        "dock_data": dock_data,
        "staging_data": staging_data,
        "shipping_data": shipping_data,
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
        "next_step": "aisle"
    }

# 4. Aisle Agent: Draws lines between rack rows
def aisle_agent(state: AgentState):
    print("--- Aisle Agent ---")
    new_messages = ["Aisle Agent: Calculating aisle paths..."]
    
    try:
        # Load from memory or state
        try:
            with open("warehouse_memory.json", "r") as f:
                memory_data = json.load(f)
                placed_racks = memory_data.get("occupied_spaces", [])
                print(f"Loaded {len(placed_racks)} racks from memory.")
        except Exception as e:
            print(f"Memory load failed: {e}")
            new_messages.append(f"Aisle Agent: Memory load skipped ({str(e)}). Using state.")
            placed_racks = [item for item in state.get('placed_items', []) if item.get('type') == 'rack']

        if not placed_racks:
            new_messages.append("Aisle Agent: No racks found to draw aisles.")
            return {"messages": new_messages}

        # Group by Z coordinate (Rows)
        rows = {}
        for r in placed_racks:
            # Ensure float
            z = float(r['z'])
            z_key = round(z, 2)
            if z_key not in rows:
                rows[z_key] = []
            rows[z_key].append(r)
        
        sorted_z = sorted(rows.keys())
        print(f"Found {len(sorted_z)} rows: {sorted_z}")
        
        aisle_lines = []
        
        # Iterate through adjacent rows
        for i in range(len(sorted_z) - 1):
            z1 = sorted_z[i]
            z2 = sorted_z[i+1]
            
            aisle_z = (z1 + z2) / 2.0
            
            row1_xs = [float(r['x']) for r in rows[z1]]
            row2_xs = [float(r['x']) for r in rows[z2]]
            
            # Combine for extent
            all_xs = row1_xs + row2_xs
            # We ideally want the 'inner' bounds or 'outer' bounds? 
            # Prompt: "line1 rack-------aiel------line2 racks"
            # Let's use the extent of the racks.
            
            # Use rack width to find edges
            # But here we just have centers in r['x']? 
            # r has 'w'.
            
            min_x_edge = 10000
            max_x_edge = -10000
            
            combined = rows[z1] + rows[z2]
            for r in combined:
                rx = float(r['x'])
                rw = float(r['w'])
                min_x_edge = min(min_x_edge, rx - rw/2)
                max_x_edge = max(max_x_edge, rx + rw/2)

            line_item = {
                "type": "aisle_line",
                "x1": float(min_x_edge),
                "z1": float(aisle_z),
                "x2": float(max_x_edge),
                "z2": float(aisle_z)
            }
            aisle_lines.append(line_item)
            new_messages.append(f"Aisle Agent: Added aisle at Z={aisle_z:.2f}")

        print(f"Generated {len(aisle_lines)} aisle lines.")
        return {
            "placed_items": aisle_lines, 
            "messages": new_messages,
            "next_step": "area"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "messages": [f"Aisle Agent Error: {str(e)}"],
            "errors": [str(e)],
            "next_step": "area"
        }

# 5. Area Agent: Determines where to place forklifts
def area_agent(state: AgentState):
    print("--- Area Agent ---")
    new_messages = ["Area Agent: Checking for forklift placement area..."]
    
    forklift_data = state.get('forklift_data', {})
    assigned_area = state.get('assigned_area')
    floor = state['floor_dims']
    dock_data = state.get('dock_data', {})
    
    zones = [] # To store staging/shipping items
    
    # --- STAGING & SHIPPING PLACEMENT ---
    # Helper to check collision with existing racks
    placed_items = state.get('placed_items', [])
    racks = [p for p in placed_items if p['type'] == 'rack']
    
    def check_collision(x, z, w, l, margin=0.5):
        min_x = x - w/2 - margin
        max_x = x + w/2 + margin
        min_z = z - l/2 - margin
        max_z = z + l/2 + margin
        
        for r in racks:
            # Rack rect
            r_min_x = r['x'] - r['w']/2
            r_max_x = r['x'] + r['w']/2
            r_min_z = r['z'] - r['l']/2
            r_max_z = r['z'] + r['l']/2
            
            if (min_x < r_max_x and max_x > r_min_x and
                min_z < r_max_z and max_z > r_min_z):
                return True # Correlation
        return False

    def find_free_spot_near_dock(w_zone, l_zone):
        # Prefer dock side.
        dock_side = dock_data.get('side', 'top').lower() if dock_data else 'top'
        # Offset from wall: 5m
        
        fw = floor['width']
        fl = floor['length']
        
        candidates = []
        
        if dock_side == 'top': # Z = -fl/2
            # Try strip at Z = -fl/2 + 5 + l_zone/2
            z_target = -fl/2 + 5.0 + l_zone/2
            # Scan X from left to right
            start_x = -fw/2 + w_zone/2 + 2
            end_x = fw/2 - w_zone/2 - 2
            candidates.append((start_x, z_target))
            candidates.append((0, z_target)) # Center
            candidates.append((end_x, z_target))
            
        elif dock_side == 'bottom': # Z = fl/2
            z_target = fl/2 - 5.0 - l_zone/2
            candidates.append((-fw/2 + w_zone/2 + 2, z_target))
            candidates.append((0, z_target))
            candidates.append((fw/2 - w_zone/2 - 2, z_target))
            
        elif dock_side == 'left': # X = -fw/2
            x_target = -fw/2 + 5.0 + w_zone/2
            candidates.append((x_target, -fl/2 + l_zone/2 + 2))
            candidates.append((x_target, 0))
            candidates.append((x_target, fl/2 - l_zone/2 - 2))
            
        elif dock_side == 'right': # X = fw/2
            x_target = fw/2 - 5.0 - w_zone/2
            candidates.append((x_target, -fl/2 + l_zone/2 + 2))
            candidates.append((x_target, 0))
            candidates.append((x_target, fl/2 - l_zone/2 - 2))
            
        # Fallback: Opposite side
        candidates.append((0, 0)) # Center of warehouse?
        
        for cand_x, cand_z in candidates:
             if not check_collision(cand_x, cand_z, w_zone, l_zone):
                 return cand_x, cand_z
        
        # If all fail, just place at (0,0) and warn? Or generic corner.
        return 0, 0

    staging = state.get('staging_data', {})
    if staging and staging.get('width'):
        sw = float(staging['width'])
        sl = float(staging['length'])
        sx, sz = find_free_spot_near_dock(sw, sl)
        zones.append({
            "type": "zone", "label": "Staging Area", "color": 0xadd8e6, # Light Blue
            "x": sx, "z": sz, "w": sw, "l": sl
        })
        new_messages.append(f"Area Agent: Placing Staging Area at ({sx:.1f}, {sz:.1f})")

    shipping = state.get('shipping_data', {})
    if shipping and shipping.get('width'):
        shw = float(shipping['width'])
        shl = float(shipping['length'])
        # Try to avoid exact same spot if possible? simple logic: existing collision check checks RACKS not ZONES.
        # Let's add previously added zones to collision check momentarily?
        # For simplicity, let's just shift shipping slightly if it collides with staging?
        # Actually, let's rely on finding a different candidate if I updated logic to check 'zones' too.
        # Quick fix: Offset shipping search or just rely on candidates finding a new spot if I run check_collision against zones too.
        
        # Add current zones to a temp list for collision
        # But check_collision only checks 'racks'. 
        
        shx, shz = find_free_spot_near_dock(shw, shl)
        # Check vs staging
        overlap = False
        for z in zones:
            if abs(z['x'] - shx) < (z['w'] + shw)/2 and abs(z['z'] - shz) < (z['l'] + shl)/2:
                overlap = True
        
        if overlap:
            # Shift along X or Z
            shx += (shw + 5.0) 
            
        zones.append({
             "type": "zone", "label": "Shipping Area", "color": 0x90ee90, # Light Green
             "x": shx, "z": shz, "w": shw, "l": shl
        })
        new_messages.append(f"Area Agent: Placing Shipping Area at ({shx:.1f}, {shz:.1f})")


    # --- FORKLIFTS ---
    # Add zones to state placed_items temporarily so forklift agent doesn't overwrite?
    # Forklift agent relies on 'assigned_area' which is separate.
    
    # We still need to assign forklift area.
    
    # Logic for forklifts:
    # If assigned_area is pre-set, use it.
    # Else, calc.
    
    if assigned_area and assigned_area.get('width') and assigned_area.get('length'):
        return {
            "messages": new_messages,
            "assigned_area": assigned_area,
            "placed_items": zones, # Add zones to placement
            "next_step": "forklift"
        }
    
    # Find spot for forklifts (that isn't staging/shipping/rack)
    # ... (existing logic for corner, but ensure no collision)
    
    w_floor = floor['width']
    l_floor = floor['length']
    area_w = 10.0
    area_l = 10.0
    
    pos_x = (w_floor/2) - (area_w/2) - 2
    pos_z = (l_floor/2) - (area_l/2) - 2
    
    assigned_area = {
        "x": pos_x,
        "z": pos_z,
        "width": area_w,
        "length": area_l
    }
    
    new_messages.append(f"Area Agent: Assigned default forklift area at X={pos_x:.1f}, Z={pos_z:.1f}")
    
    return {
        "assigned_area": assigned_area,
        "placed_items": zones, # Append zones
        "messages": new_messages,
        "next_step": "forklift"
    }

# 6. Forklift Agent: Places forklifts
def forklift_agent(state: AgentState):
    print("--- Forklift Agent ---")
    new_messages = ["Forklift Agent: Placing forklifts..."]
    
    forklift_data = state.get('forklift_data', {})
    area = state.get('assigned_area')
    
    if not forklift_data or not area:
         return {"messages": ["Forklift Agent: Missing data."], "next_step": "end"}
         
    count = int(forklift_data.get('count', 1))
    fw = float(forklift_data.get('width', 1.5))
    fl = float(forklift_data.get('length', 2.5))
    fh = float(forklift_data.get('height', 2.5))
    
    items = []
    
    # Simple packing in the area
    # Grid:
    cols = int(area['width'] // (fw + 1))
    if cols < 1: cols = 1
    
    start_x = area['x'] - area['width']/2 + fw/2 + 0.5
    start_z = area['z'] - area['length']/2 + fl/2 + 0.5
    
    current_x = start_x
    current_z = start_z
    
    for i in range(count):
        item = {
            "type": "forklift",
            "id": f"forklift_{i+1}",
            "x": current_x,
            "z": current_z,
            "w": fw,
            "l": fl,
            "h": fh
        }
        items.append(item)
        
        # Move next
        current_x += fw + 1.0
        if current_x > area['x'] + area['width']/2 - fw/2:
            current_x = start_x
            current_z += fl + 1.0
            
    new_messages.append(f"Forklift Agent: Placed {count} forklifts.")
        
    return {
        "placed_items": items,
        "messages": new_messages,
        "next_step": "dock"
    }

# 7. Dock Agent: Places dock doors
def dock_agent(state: AgentState):
    print("--- Dock Agent ---")
    new_messages = ["Dock Agent: Checking for dock doors..."]
    
    dock_data = state.get('dock_data', {})
    if not dock_data or not dock_data.get('count', 0):
        new_messages.append("Dock Agent: No docks requested.")
        # Even if no docks, we should finalize memory here as it's the last step
        _save_memory(state, [])
        return {"messages": new_messages, "next_step": "end"}
        
    side = dock_data.get('side', 'top').lower()
    count = int(dock_data.get('count', 1))
    width = float(dock_data.get('width', 3.0)) # Door width
    
    floor = state['floor_dims']
    fw = floor['width']
    fl = floor['length']
    fh = floor['height']
    
    items = []
    
    # Determine Wall Coordinate
    # Top: Z = -Length/2
    # Bottom: Z = Length/2
    # Left: X = -Width/2
    # Right: X = Width/2
    
    # We distribute them along the chosen wall.
    # Spacing = WallLength / (Count + 1)
    
    if side == 'top':
        wall_len = fw
        z_pos = -fl/2
        spacing = wall_len / (count + 1)
        for i in range(count):
            x = -fw/2 + spacing * (i+1)
            item = {"type": "dock_door", "x": x, "z": z_pos, "w": width, "h": fh, "rot": 0, "side": "top"}
            items.append(item)
            
    elif side == 'bottom':
        wall_len = fw
        z_pos = fl/2
        spacing = wall_len / (count + 1)
        for i in range(count):
            x = -fw/2 + spacing * (i+1)
            item = {"type": "dock_door", "x": x, "z": z_pos, "w": width, "h": fh, "rot": 0, "side": "bottom"}
            items.append(item)
            
    elif side == 'left':
        wall_len = fl
        x_pos = -fw/2
        spacing = wall_len / (count + 1)
        for i in range(count):
            z = -fl/2 + spacing * (i+1)
            item = {"type": "dock_door", "x": x_pos, "z": z, "w": width, "h": fh, "rot": 90, "side": "left"}
            items.append(item)
            
    elif side == 'right':
        wall_len = fl
        x_pos = fw/2
        spacing = wall_len / (count + 1)
        for i in range(count):
            z = -fl/2 + spacing * (i+1)
            item = {"type": "dock_door", "x": x_pos, "z": z, "w": width, "h": fh, "rot": 90, "side": "right"}
            items.append(item)
            
    new_messages.append(f"Dock Agent: Placed {count} docks on {side} wall.")
    
    # --- Finalize Memory ---
    _save_memory(state, items)
    
    return {
        "placed_items": items,
        "messages": new_messages,
        "next_step": "end"
    }

def _save_memory(state: AgentState, new_items: List[Dict]):
    """Helper to save absolute final state to memory"""
    try:
        # Load existing or start fresh
        # Actually state['placed_items'] should hold everything accumulated if we set operator.add
        # But let's look at what's in state.
        
        # We need to list EVERYTHING.
        # state['placed_items'] likely has [floor, walls, ...racks..., ...aisles..., ...forklifts...]
        # But 'new_items' are not yet in state when this runs? 
        # operator.add merges result of this function into state. So they are not there yet.
        
        all_items = state.get('placed_items', []) + new_items
        
        # Calculate stats
        floor = state['floor_dims']
        floor_area = floor['width'] * floor['length']
        
        occupied_area = 0
        for item in all_items:
            if item.get('type') in ['rack']:
                occupied_area += item.get('occupied_area', 0)
            elif item.get('type') == 'forklift':
                occupied_area += (item.get('w', 0) * item.get('l', 0))
            elif item.get('type') == 'zone':
                occupied_area += (item.get('w', 0) * item.get('l', 0))
                
        free_area = floor_area - occupied_area
         
        memory_snapshot = {
            "floor": floor,
            "stats": {
                "total_area": floor_area,
                "occupied_area": occupied_area,
                "free_area": free_area,
                "utilization": f"{(occupied_area/floor_area)*100:.2f}%"
            },
            "occupied_spaces": all_items
        }
        
        with open("warehouse_memory.json", "w") as f:
            json.dump(memory_snapshot, f, indent=2)
            
        print("Dock Agent: Memory snapshot updated with all items.")
        
    except Exception as e:
        print(f"Error saving memory: {e}")


# --- Graph ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_agent)
    workflow.add_node("floor", floor_agent)
    workflow.add_node("racks", rack_agent)
    workflow.add_node("aisle", aisle_agent)
    workflow.add_node("area", area_agent)
    workflow.add_node("forklift", forklift_agent)
    workflow.add_node("dock", dock_agent)
    
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
    workflow.add_edge("racks", "aisle")
    workflow.add_edge("aisle", "area")
    
    def router_area(state):
        return state['next_step']
        
    workflow.add_conditional_edges("area", router_area, {"forklift": "forklift", "end": "dock"})
    workflow.add_edge("forklift", "dock")
    workflow.add_edge("dock", END)
    
    return workflow.compile()

agent_graph = build_graph()
