# Warehouse_Agents

A multi-agent system for warehouse planning and optimization using LangGraph and Google GenAI.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JapanD8/Warehouse_Agents.git
    cd Warehouse_Agents
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## Running the Application

1.  **Start the server:**
    ```bash
    python server.py
    ```
    The server will start at `http://0.0.0.0:8001`.

2.  **Access the Interface:**
    Open your browser and navigate to:
    [http://localhost:8001](http://localhost:8001)

## Usage

-   Upload a CSV file containing warehouse data.
-   The agents will process the data and generate a layout plan.
-   You can view the results on the web interface.
