
// Main Application Logic

document.addEventListener('DOMContentLoaded', () => {
    // 1. Init 3D Scene
    initScene('canvas-container');

    // 2. Setup WebSocket
    const ws = new WebSocket(`ws://${location.host}/ws`);

    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const chatContainer = document.getElementById('chat-container');
    const fileInput = document.getElementById('csvFile');

    ws.onopen = () => {
        statusText.textContent = "Connected";
        statusDot.classList.add('active');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };

    ws.onclose = () => {
        statusText.textContent = "Disconnected";
        statusDot.classList.remove('active');
    };

    // 3. File Upload Handler
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (evt) => {
            const csvContent = evt.target.result;
            addChatMessage("User", `Uploaded ${file.name}. Starting agents...`, 'msg-user');

            // Clear previous scene
            clearScene();

            // Send to backend
            ws.send(JSON.stringify({
                type: 'run_plan',
                csv: csvContent
            }));
        };
        reader.readAsText(file);
    });

    // Helper to add chat messages
    function addChatMessage(agent, text, className) {
        const div = document.createElement('div');
        div.className = `message ${className}`;
        div.innerHTML = `<strong>${agent}:</strong> ${text}`;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Handle updates from backend
    function handleServerMessage(data) {
        if (data.type === 'update') {
            // Update Chat
            if (data.messages && data.messages.length > 0) {
                // Determine agent style class
                let style = 'msg-planner';
                if (data.node === 'floor') style = 'msg-floor';
                if (data.node === 'racks') style = 'msg-rack';

                // Only show the *latest* message from the chunk to avoid spam if entire history is sent
                // But our backend sends the delta usually? 
                // Wait, in server.py: "messages": state_update.get("messages", [])
                // "state_update" from LangGraph usually contains the whole list if it's accumulating?
                // No, standard nodes usually return the *update*. 
                // Let's assume the list might be full or partial.
                // To be safe, we might just print the last one if it's a list.
                // Or better, let's just print them all if they are new.
                // We'll trust the backend sends new relevant info.

                data.messages.forEach(msg => {
                    // Simple de-duping if needed, but let's just append
                    addChatMessage(data.node.toUpperCase(), msg, style);
                });
            }

            // Update 3D Scene
            if (data.items && data.items.length > 0) {
                updateScene(data.items);
            }
        }
        else if (data.type === 'complete') {
            addChatMessage("SYSTEM", "Plan execution finished.", "msg-planner");
        }
    }
});
