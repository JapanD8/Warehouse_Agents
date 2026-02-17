
// Three.js Scene Logic
const sceneState = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    floorGroup: null,
    rackGroup: null,
    gridHelper: null
};

// Initialize the 3D Scene
function initScene(containerId) {
    const container = document.getElementById(containerId);

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f1115);
    scene.fog = new THREE.FogExp2(0x0f1115, 0.002);

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(50, 60, 50);
    camera.lookAt(0, 0, 0);

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 4. Controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; // Smooth!
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 10;
    controls.maxDistance = 200;
    controls.maxPolarAngle = Math.PI / 2 - 0.1; // Don't go below floor

    // 5. Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 0.7);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(50, 100, 50);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 500;

    const d = 100;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d;
    dirLight.shadow.camera.bottom = -d;
    scene.add(dirLight);

    // 6. Helpers
    const gridHelper = new THREE.GridHelper(200, 100, 0x1e293b, 0x1e293b);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // 7. Groups
    const floorGroup = new THREE.Group();
    const rackGroup = new THREE.Group();
    scene.add(floorGroup);
    scene.add(rackGroup);

    // State storage
    sceneState.scene = scene;
    sceneState.camera = camera;
    sceneState.renderer = renderer;
    sceneState.controls = controls;
    sceneState.floorGroup = floorGroup;
    sceneState.rackGroup = rackGroup;
    sceneState.gridHelper = gridHelper;

    // Resize Handler
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update(); // Required for damping
        renderer.render(scene, camera);
    }
    animate();
}

function clearScene() {
    // Remove children from groups properly
    while (sceneState.floorGroup.children.length > 0) {
        sceneState.floorGroup.remove(sceneState.floorGroup.children[0]);
    }
    while (sceneState.rackGroup.children.length > 0) {
        sceneState.rackGroup.remove(sceneState.rackGroup.children[0]);
    }
}

function updateScene(items) {
    if (!items || items.length === 0) return;

    items.forEach(item => {
        if (item.type === 'floor') {
            buildFloor(item);
        } else if (item.type === 'rack') {
            buildRack(item);
        } else if (item.type === 'walls') {
            buildWalls(item);
        } else if (item.type === 'aisle_line') {
            buildAisle(item);
        } else if (item.type === 'forklift') {
            buildForklift(item);
        }
    });
}


function buildFloor(data) {
    // Check if floor already exists to avoid dupes (though clearScene should handle it)
    if (sceneState.floorGroup.children.find(c => c.name === 'main_floor')) return;

    const w = data.width;
    const l = data.length;

    const geo = new THREE.PlaneGeometry(w, l);
    const mat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        roughness: 0.8,
        metalness: 0.2,
        side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.receiveShadow = true;
    mesh.name = 'main_floor';

    sceneState.floorGroup.add(mesh);

    // Update grid helper? Nah, keeps context.
}

function buildWalls(data) {
    // Visual-only walls
    const w = data.width;
    const l = data.length;
    const h = data.height;

    const mat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide
    });

    const wallGeo = new THREE.BoxGeometry(w, h, l);
    // Invert normals? Or just use wireframe helper
    const edges = new THREE.EdgesGeometry(wallGeo);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x475569 }));
    line.position.y = h / 2;

    sceneState.floorGroup.add(line);
}

function buildRack(data) {
    // data: { x, z, w, l, h, id }
    const geo = new THREE.BoxGeometry(data.w, data.h, data.l);
    const mat = new THREE.MeshStandardMaterial({ color: 0x3b82f6 });
    const mesh = new THREE.Mesh(geo, mat);

    mesh.position.set(data.x, data.h / 2, data.z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    // Add black edges for visibility
    const edges = new THREE.EdgesGeometry(geo);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x1e3a8a }));
    mesh.add(line);

    sceneState.rackGroup.add(mesh);
}

function buildAisle(data) {
    // data: { x1, z1, x2, z2 }
    console.log("Building aisle:", data);

    const dx = data.x2 - data.x1;
    const dz = data.z2 - data.z1;
    const length = Math.sqrt(dx * dx + dz * dz);

    if (length <= 0) return;

    const cx = (data.x1 + data.x2) / 2;
    const cz = (data.z1 + data.z2) / 2;

    // Width (X) = length, Height (Y) = 0.05, Depth (Z) = 0.5
    const geo = new THREE.BoxGeometry(length, 0.05, 0.5);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffff00 });
    const mesh = new THREE.Mesh(geo, mat);

    mesh.position.set(cx, 0.1, cz);

    // Rotate if needed (for now assumes horizontal aisles along X)
    const angle = Math.atan2(dz, dx);
    mesh.rotation.y = -angle;

    mesh.name = 'aisle_strip';
    sceneState.floorGroup.add(mesh);
}

function buildForklift(data) {
    // data: { x, z, w, l, h, id }
    // Draw a simple forklift proxy (e.g., orange box with a smaller box on front)

    const group = new THREE.Group();
    group.position.set(data.x, 0, data.z);

    // Body
    const bodyGeo = new THREE.BoxGeometry(data.w, data.h * 0.6, data.l);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0xffa500 }); // Orange
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = (data.h * 0.6) / 2;
    body.castShadow = true;
    group.add(body);

    // Mast/Forks (Visual indication)
    const mastGeo = new THREE.BoxGeometry(data.w * 0.1, data.h, data.l * 0.1);
    const mastMat = new THREE.MeshStandardMaterial({ color: 0x333333 });
    const mast = new THREE.Mesh(mastGeo, mastMat);
    mast.position.set(data.w / 2, data.h / 2, 0); # Front ?
        group.add(mast);

    sceneState.rackGroup.add(group); # Add to rack group(or create new items group)
}
