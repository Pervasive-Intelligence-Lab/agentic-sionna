import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ─── State ───────────────────────────────────────────
const state = {
    view: 'indoor',
    sidebarTab: 'scene',
    layers: { heatmap: true, walls: true, furniture: true, tx: true, rays: false },
    realistic: false,
    currentScene: null,
    coverageData: null,
    coverageSlices: null,
    rayData: null,
    rayAnimating: false,
    drillBuilding: null,
    selectedBuilding: null,  // building data when clicked in outdoor view
    catalogCategories: null,
    glbCache: {},        // model_id -> THREE.Object3D (template)
    antennaConfig: { pattern: 'tr38901', polarization: 'cross', rows: 8, cols: 8, azimuth: 0, elevation: 0 },
    activeEventSource: null,
};

let FURNITURE_CATEGORIES = [
    'bed','desk','chair','sofa','wardrobe','nightstand',
    'bookcase','cabinet','table','coffee_table','tv_stand','lamp'
];

const MATERIAL_COLORS = {
    wood: 0x8B5A2B,
    metal: 0xA0AAB4,
    glass: 0x64B4FF,
    concrete: 0x888888,
};

const MATERIAL_COLORS_HEX = {
    wood: '#8B5A2B',
    metal: '#A0AAB4',
    glass: '#64B4FF',
    concrete: '#888888',
};

const RAY_COLORS = [0xFF3B30, 0xFF6B30, 0xFF9500, 0xFFCC00, 0x4CD964, 0x007AFF];

// ─── Furniture interaction state machine ─────────────
let handleGroup = null; // THREE.Group on scene3d, NOT in layerGroups (survives clearAllGroups)
const FurnInteraction = { IDLE: 'idle', SELECTED: 'selected', DRAGGING_MOVE: 'dragging_move', DRAGGING_ROTATE: 'dragging_rotate' };
let furnInteractionState = FurnInteraction.IDLE;
let selectedFurnIdx = -1;
let selectedFurnMeshRef = null; // direct ref to THREE mesh for perf during drag
let dragStartPointer = { x: 0, y: 0 };
let dragStartFurnPos = { x: 0, y: 0 };
let dragStartTheta = 0;
let dragStartAngle = 0;

// ─── Undo/Redo stack for furniture layout ────────────
const undoStack = [];
const redoStack = [];
const MAX_UNDO = 50;

function snapshotFurniture() {
    const room = state.currentScene && state.currentScene.room;
    if (!room || !room.furniture) return null;
    return JSON.parse(JSON.stringify(room.furniture));
}

function pushUndo() {
    const snap = snapshotFurniture();
    if (!snap) return;
    undoStack.push(snap);
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    redoStack.length = 0; // clear redo on new action
}

function performUndo() {
    const room = state.currentScene && state.currentScene.room;
    if (!room || undoStack.length === 0) return;
    redoStack.push(snapshotFurniture());
    room.furniture = undoStack.pop();
    deselectFurniture();
    updatePlacedFurnitureList();
    saveFurniture();
}

function performRedo() {
    const room = state.currentScene && state.currentScene.room;
    if (!room || redoStack.length === 0) return;
    undoStack.push(snapshotFurniture());
    room.furniture = redoStack.pop();
    deselectFurniture();
    updatePlacedFurnitureList();
    saveFurniture();
}

function saveFurniture() {
    const sc = state.currentScene;
    if (!sc || !sc.scene_id) return;
    const room = sc.room;
    if (!room || !room.furniture) return;
    fetch(`/api/scenes/${sc.scene_id}/furniture`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ furniture: room.furniture }),
    }).catch(() => {}); // fire-and-forget
}

const FURNITURE_ICONS = {
    bed: '\u{1F6CF}', desk: '\u{1F4DD}', chair: '\u{1FA91}', sofa: '\u{1F6CB}',
    wardrobe: '\u{1F6AA}', nightstand: '\u{1F4A1}', bookcase: '\u{1F4DA}',
    cabinet: '\u{1F5C4}', table: '\u{1F37D}', coffee_table: '\u2615',
    tv_stand: '\u{1F4FA}', lamp: '\u{1F4A1}',
};

function getFurnitureIcon(cat) {
    const key = (cat || '').toLowerCase().replace(/\s+/g, '_');
    return FURNITURE_ICONS[key] || '\u{1F4E6}';
}

// ─── Plotly layouts (analysis panels only) ───────────
const smallLayout = {
    paper_bgcolor: '#111',
    plot_bgcolor: '#1a1a1a',
    font: { color: '#9E9E9E', size: 9 },
    margin: { l: 30, r: 10, t: 5, b: 25 },
    showlegend: false,
    xaxis: { gridcolor: '#262626' },
    yaxis: { gridcolor: '#262626' },
};

// ─── Three.js Setup ──────────────────────────────────
let renderer, camera, controls, scene3d;
const layerGroups = {};
const gltfLoader = new GLTFLoader();
let savedCameraPos = null, savedControlsTarget = null;

function initThreeViewport() {
    const canvas = document.getElementById('viewport-canvas');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setClearColor(0x050505);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // Three.js r155+ uses physical light intensity (lux/candela), so the
    // old 1.0-ish values render nearly-black under ACES tonemapping. Use
    // linear (no tonemap) + sRGB output, and bump lights up below.
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    scene3d = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
    camera.position.set(6, 6, 5);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.target.set(2.5, 0, 2);

    // Prevent browser context menu on canvas so right-click pan works
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    // Lights — bumped for r170 physical-intensity mode. Ambient carries most
    // of the fill so textured furniture reads its colours; two directionals
    // give some shape.
    const ambient = new THREE.AmbientLight(0xffffff, 3.0);
    scene3d.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 2.5);
    dirLight.position.set(10, 20, 10);
    const fillLight = new THREE.DirectionalLight(0xffffff, 1.5);
    fillLight.position.set(-10, 10, -10);
    scene3d.add(dirLight);
    scene3d.add(fillLight);
    // A hemisphere light gives sky/ground bounce so ceiling-facing surfaces
    // don't crush to black.
    const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 1.5);
    scene3d.add(hemi);

    // Layer groups
    for (const name of ['floor', 'walls', 'furniture', 'heatmap', 'tx', 'rays', 'buildings', 'roads', 'trees']) {
        layerGroups[name] = new THREE.Group();
        layerGroups[name].name = name;
        scene3d.add(layerGroups[name]);
    }

    // Handle group for furniture manipulation (survives clearAllGroups)
    handleGroup = new THREE.Group();
    handleGroup.name = '__handles__';
    scene3d.add(handleGroup);

    // Responsive resize
    const ro = new ResizeObserver(() => {
        const parent = canvas.parentElement;
        const w = parent.clientWidth;
        const h = parent.clientHeight;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    });
    ro.observe(canvas.parentElement);
    // Initial size
    const parent = canvas.parentElement;
    renderer.setSize(parent.clientWidth, parent.clientHeight);
    camera.aspect = parent.clientWidth / parent.clientHeight;
    camera.updateProjectionMatrix();

    // Raycaster for building hover + click-to-select
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const tooltip = document.getElementById('building-tooltip');
    let hoveredBuildingMesh = null;

    // ─── Furniture interaction: floor plane + raycasting ──
    const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersectPt = new THREE.Vector3();

    // ─── Furniture pointer events (select → handle drag) ──
    canvas.addEventListener('pointerdown', (e) => {
        canvas._pointerDownTime = Date.now();
        dragStartPointer.x = e.clientX;
        dragStartPointer.y = e.clientY;

        if (state.view !== 'indoor' || !state.currentScene || !state.currentScene.room) return;
        if (e.button !== 0) return;

        const rect = canvas.getBoundingClientRect();
        pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);

        if (furnInteractionState === FurnInteraction.SELECTED) {
            // Check rotation ring first
            if (handleGroup.children.length > 0) {
                const handleHits = raycaster.intersectObjects(handleGroup.children, true);
                if (handleHits.length > 0) {
                    let obj = handleHits[0].object;
                    while (obj && !obj.userData.handleType) obj = obj.parent;
                    if (obj && obj.userData.handleType === 'rotate') {
                        pushUndo();
                        furnInteractionState = FurnInteraction.DRAGGING_ROTATE;
                        controls.enabled = false;
                        const furn = state.currentScene.room.furniture[selectedFurnIdx];
                        dragStartTheta = furn.theta || 0;
                        raycaster.ray.intersectPlane(floorPlane, intersectPt);
                        dragStartAngle = Math.atan2(intersectPt.z - furn.y, intersectPt.x - furn.x);
                        canvas.style.cursor = 'grabbing';
                        e.preventDefault();
                        return;
                    }
                }
            }

            // Check if clicking on the selected furniture mesh → start move drag
            // But exclude clicks in the gap between furniture edge and rotation ring
            const furnHits = raycaster.intersectObjects(layerGroups.furniture.children, true);
            if (furnHits.length > 0) {
                let obj = furnHits[0].object;
                while (obj && !obj.userData.id && obj.parent && obj.parent !== layerGroups.furniture) {
                    obj = obj.parent;
                }
                if (obj && obj.userData.id) {
                    const room = state.currentScene.room;
                    const hitIdx = room.furniture.findIndex(f => f.id === obj.userData.id);
                    if (hitIdx === selectedFurnIdx) {
                        // Verify hit is actually on the furniture body (Y > 0.05), not
                        // a grazing floor-level ray that clips through the box base
                        const hitPt = furnHits[0].point;
                        if (hitPt.y < 0.05) {
                            // Floor-level hit — likely in the ring gap area, ignore
                            return;
                        }
                        pushUndo();
                        furnInteractionState = FurnInteraction.DRAGGING_MOVE;
                        controls.enabled = false;
                        const furn = room.furniture[selectedFurnIdx];
                        dragStartFurnPos.x = furn.x;
                        dragStartFurnPos.y = furn.y;
                        raycaster.ray.intersectPlane(floorPlane, intersectPt);
                        dragStartPointer._floorX = intersectPt.x;
                        dragStartPointer._floorZ = intersectPt.z;
                        canvas.style.cursor = 'grabbing';
                        e.preventDefault();
                        return;
                    }
                }
            }
        }
    });

    canvas.addEventListener('pointermove', (e) => {
        if (state.view !== 'indoor' || !state.currentScene || !state.currentScene.room) {
            // Outdoor building hover (unchanged)
            if (state.view === 'outdoor' && !state.drillBuilding) {
                const rect = canvas.getBoundingClientRect();
                pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                raycaster.setFromCamera(pointer, camera);
                const hits = raycaster.intersectObjects(layerGroups.buildings.children, true);
                if (hits.length > 0) {
                    let obj = hits[0].object;
                    while (obj && !obj.userData.buildingData) obj = obj.parent;
                    if (obj && obj.userData.buildingData) {
                        const bd = obj.userData.buildingData;
                        const label = bd.name || bd.id;
                        tooltip.textContent = `${label}  (${bd.h.toFixed(0)}m, ${bd.material})`;
                        tooltip.style.display = 'block';
                        tooltip.style.left = (e.clientX - canvas.getBoundingClientRect().left + 14) + 'px';
                        tooltip.style.top = (e.clientY - canvas.getBoundingClientRect().top - 10) + 'px';
                        canvas.style.cursor = 'pointer';
                        if (hoveredBuildingMesh && hoveredBuildingMesh !== obj && hoveredBuildingMesh !== state._selectedBuildingMesh) {
                            hoveredBuildingMesh.material.emissive.setHex(0x000000);
                        }
                        if (obj !== state._selectedBuildingMesh) {
                            obj.material.emissive.setHex(0x222222);
                        }
                        hoveredBuildingMesh = obj;
                        return;
                    }
                }
                tooltip.style.display = 'none';
                if (furnInteractionState === FurnInteraction.IDLE) canvas.style.cursor = 'default';
                if (hoveredBuildingMesh && hoveredBuildingMesh !== state._selectedBuildingMesh) {
                    hoveredBuildingMesh.material.emissive.setHex(0x000000);
                    hoveredBuildingMesh = null;
                }
            }
            return;
        }

        const rect = canvas.getBoundingClientRect();
        pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);

        if (furnInteractionState === FurnInteraction.DRAGGING_MOVE) {
            const room = state.currentScene.room;
            const furn = room.furniture[selectedFurnIdx];
            if (!furn) return;
            if (raycaster.ray.intersectPlane(floorPlane, intersectPt)) {
                const dx = intersectPt.x - dragStartPointer._floorX;
                const dz = intersectPt.z - dragStartPointer._floorZ;
                // Rotated axis-aligned bounding box half-extents
                const theta = (furn.theta || 0) * Math.PI / 180;
                const cosT = Math.abs(Math.cos(theta)), sinT = Math.abs(Math.sin(theta));
                const hw = cosT * furn.width / 2 + sinT * furn.depth / 2;
                const hd = sinT * furn.width / 2 + cosT * furn.depth / 2;
                furn.x = Math.max(hw, Math.min(room.width - hw, dragStartFurnPos.x + dx));
                furn.y = Math.max(hd, Math.min(room.length - hd, dragStartFurnPos.y + dz));
                // Direct mesh update (no full redraw)
                if (selectedFurnMeshRef) {
                    selectedFurnMeshRef.position.x = furn.x;
                    selectedFurnMeshRef.position.z = furn.y;
                }
                handleGroup.position.set(furn.x, 0, furn.y);
            }
            return;
        }

        if (furnInteractionState === FurnInteraction.DRAGGING_ROTATE) {
            const furn = state.currentScene.room.furniture[selectedFurnIdx];
            if (!furn) return;
            if (raycaster.ray.intersectPlane(floorPlane, intersectPt)) {
                const currentAngle = Math.atan2(intersectPt.z - furn.y, intersectPt.x - furn.x);
                const delta = (currentAngle - dragStartAngle) * (180 / Math.PI);
                furn.theta = (dragStartTheta + delta) % 360;
                // Direct mesh update
                if (selectedFurnMeshRef) {
                    selectedFurnMeshRef.rotation.y = -(furn.theta * Math.PI / 180);
                }
                handleGroup.rotation.y = -(furn.theta * Math.PI / 180);
            }
            return;
        }

        // Hover cursor feedback when SELECTED
        if (furnInteractionState === FurnInteraction.SELECTED) {
            // Check rotation ring
            if (handleGroup.children.length > 0) {
                const handleHits = raycaster.intersectObjects(handleGroup.children, true);
                if (handleHits.length > 0) {
                    canvas.style.cursor = 'ew-resize';
                    return;
                }
            }
            // Check furniture mesh (for move)
            const furnHits = raycaster.intersectObjects(layerGroups.furniture.children, true);
            if (furnHits.length > 0) {
                let obj = furnHits[0].object;
                while (obj && !obj.userData.id && obj.parent && obj.parent !== layerGroups.furniture) {
                    obj = obj.parent;
                }
                if (obj && obj.userData.id) {
                    const room = state.currentScene.room;
                    const hitIdx = room.furniture.findIndex(f => f.id === obj.userData.id);
                    if (hitIdx === selectedFurnIdx) {
                        canvas.style.cursor = 'grab';
                        return;
                    }
                }
            }
            canvas.style.cursor = 'default';
            return;
        }

        // IDLE: check if hovering furniture for pointer cursor
        if (furnInteractionState === FurnInteraction.IDLE) {
            const furnHits = raycaster.intersectObjects(layerGroups.furniture.children, true);
            canvas.style.cursor = furnHits.length > 0 ? 'pointer' : 'default';
        }
    });

    canvas.addEventListener('pointerup', (e) => {
        // ─── Outdoor building click ───
        if (state.view === 'outdoor' && !state.drillBuilding) {
            if (Date.now() - (canvas._pointerDownTime || 0) > 300) return;
            const rect = canvas.getBoundingClientRect();
            pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(pointer, camera);
            const hits = raycaster.intersectObjects(layerGroups.buildings.children, true);
            if (hits.length > 0) {
                let obj = hits[0].object;
                while (obj && !obj.userData.buildingData) obj = obj.parent;
                if (obj && obj.userData.buildingData) {
                    selectBuilding(obj, obj.userData.buildingData);
                    return;
                }
            }
            deselectBuilding();
            return;
        }

        if (state.view !== 'indoor' || !state.currentScene || !state.currentScene.room) return;

        // ─── Finish dragging → back to SELECTED ───
        if (furnInteractionState === FurnInteraction.DRAGGING_MOVE || furnInteractionState === FurnInteraction.DRAGGING_ROTATE) {
            controls.enabled = true;
            furnInteractionState = FurnInteraction.SELECTED;
            canvas.style.cursor = 'default';
            drawSceneViewport();
            updatePlacedFurnitureList();
            saveFurniture();
            return;
        }

        // ─── Click detection (not a drag) ───
        const dx = e.clientX - dragStartPointer.x;
        const dy = e.clientY - dragStartPointer.y;
        if (Math.sqrt(dx * dx + dy * dy) > 5) return; // was a drag/orbit, not a click

        const rect = canvas.getBoundingClientRect();
        pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);

        // Raycast furniture
        const furnHits = raycaster.intersectObjects(layerGroups.furniture.children, true);
        if (furnHits.length > 0) {
            let obj = furnHits[0].object;
            while (obj && !obj.userData.id && obj.parent && obj.parent !== layerGroups.furniture) {
                obj = obj.parent;
            }
            if (obj && obj.userData.id) {
                const room = state.currentScene.room;
                const idx = room.furniture.findIndex(f => f.id === obj.userData.id);
                if (idx >= 0) {
                    selectFurniture(idx);
                    return;
                }
            }
        }

        // Clicked empty space → deselect
        if (furnInteractionState !== FurnInteraction.IDLE) {
            deselectFurniture();
        }
    });

    // Keyboard controls for furniture interaction + undo/redo
    document.addEventListener('keydown', (e) => {
        // Undo/Redo: Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y) — works in any state
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            performUndo();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === 'Z' || e.key === 'y')) {
            e.preventDefault();
            performRedo();
            return;
        }

        if (furnInteractionState === FurnInteraction.IDLE) return;
        if (e.key === 'Escape') {
            deselectFurniture();
        } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedFurnIdx >= 0) {
            const room = state.currentScene && state.currentScene.room;
            if (!room || !room.furniture) return;
            pushUndo();
            room.furniture.splice(selectedFurnIdx, 1);
            deselectFurniture(); // calls drawSceneViewport internally
            updatePlacedFurnitureList();
            saveFurniture();
        }
    });

    // Click-away to dismiss chat history (only relevant when chat-history
    // exists as a separate popover — with the right-side chat column it's
    // gone, so this becomes a no-op safely).
    document.addEventListener('pointerdown', (e) => {
        const chatHistory = document.getElementById('chat-history');
        const chatInline = document.getElementById('chat-inline');
        if (!chatHistory || !chatInline) return;
        if (!chatHistory.classList.contains('open')) return;
        if (chatHistory.contains(e.target) || chatInline.contains(e.target)) return;
        chatHistory.classList.remove('open');
    });

    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene3d, camera);
    }
    animate();
}

// ─── Coordinate mapping ──────────────────────────────
// Data: X=east, Y=north, Z=up  (Z-up)
// Three.js: X=right, Y=up, Z=forward (Y-up)
// Mapping: data.x → three.x, data.y → three.z, data.z → three.y
function toThree(dx, dy, dz) {
    return new THREE.Vector3(dx, dz, dy);
}

// ─── Clear helpers ───────────────────────────────────
function clearGroup(group) {
    while (group.children.length > 0) {
        const child = group.children[0];
        group.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
            if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
            else child.material.dispose();
        }
    }
}

function clearAllGroups() {
    for (const g of Object.values(layerGroups)) clearGroup(g);
}

// ─── Camera save/restore ─────────────────────────────
function saveCamera() {
    if (camera) {
        savedCameraPos = camera.position.clone();
        savedControlsTarget = controls.target.clone();
    }
}

function restoreCamera() {
    if (savedCameraPos) {
        camera.position.copy(savedCameraPos);
        controls.target.copy(savedControlsTarget);
        savedCameraPos = null;
        savedControlsTarget = null;
    }
}

// ─── Indoor Scene Builders ───────────────────────────

function buildFloor(w, l, polygon) {
    let geom;
    if (polygon && polygon.length >= 3) {
        // Polygon floor: negate Y before rotation (same trick as buildings)
        const shape = new THREE.Shape();
        shape.moveTo(polygon[0][0], -polygon[0][1]);
        for (let i = 1; i < polygon.length; i++) shape.lineTo(polygon[i][0], -polygon[i][1]);
        shape.closePath();
        geom = new THREE.ShapeGeometry(shape);
    } else {
        geom = new THREE.PlaneGeometry(w, l);
        // Negate Y so after rotateX(-PI/2): (w/2, -l/2, 0) → (w/2, 0, l/2) ✓
        geom.translate(w / 2, -l / 2, 0);
    }
    // rotateX(-PI/2) maps (x, y, z) → (x, z, -y)
    // With negated Y: (x, -y, 0) → (x, 0, y) — correct XZ floor plane
    geom.rotateX(-Math.PI / 2);

    const mat = new THREE.MeshStandardMaterial({
        color: state.realistic ? 0x8B7355 : 0x2a2a2a,
        roughness: 0.9,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: state.realistic ? 0.55 : 0.45,
    });
    const mesh = new THREE.Mesh(geom, mat);
    return mesh;
}

function buildWalls(w, l, h, polygon) {
    const group = new THREE.Group();

    let corners;
    if (polygon && polygon.length >= 3) {
        corners = polygon;
    } else {
        corners = [[0, 0], [w, 0], [w, l], [0, l]];
    }

    if (state.realistic) {
        // Realistic mode: semi-transparent wall panels, FrontSide only
        // FrontSide = only the inward-facing side renders, so walls
        // behind the camera (facing away) are invisible → never blocks view
        const wallMat = new THREE.MeshStandardMaterial({
            color: 0xE8E0D0,
            transparent: true,
            opacity: 0.10,
            side: THREE.FrontSide,
            roughness: 0.85,
            depthWrite: false,
        });

        for (let i = 0; i < corners.length; i++) {
            const p0 = corners[i];
            const p1 = corners[(i + 1) % corners.length];
            const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
            const len = Math.sqrt(dx * dx + dy * dy);
            if (len < 0.001) continue;

            const geom = new THREE.PlaneGeometry(len, h);
            const wall = new THREE.Mesh(geom, wallMat.clone());
            const cx = (p0[0] + p1[0]) / 2;
            const cy = (p0[1] + p1[1]) / 2;
            wall.position.set(cx, h / 2, cy);
            // PlaneGeometry faces +Z by default. Rotate so it faces inward.
            // Wall segment angle + PI to face inward
            const angle = Math.atan2(dy, dx);
            wall.rotation.y = -angle + Math.PI;
            group.add(wall);
        }
    }

    // Always draw wireframe edges (subtle in realistic, primary in wireframe)
    const edgeColor = state.realistic ? 0x887766 : 0xC0B8A8;
    const lineMat = new THREE.LineBasicMaterial({ color: edgeColor, linewidth: 1 });

    for (let i = 0; i < corners.length; i++) {
        const p0 = corners[i];
        const p1 = corners[(i + 1) % corners.length];

        // Vertical edges at each corner
        const verts = new Float32Array([p0[0], 0, p0[1], p0[0], h, p0[1]]);
        const vGeom = new THREE.BufferGeometry();
        vGeom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        group.add(new THREE.Line(vGeom, lineMat));

        // Top edge
        const tVerts = new Float32Array([p0[0], h, p0[1], p1[0], h, p1[1]]);
        const tGeom = new THREE.BufferGeometry();
        tGeom.setAttribute('position', new THREE.BufferAttribute(tVerts, 3));
        group.add(new THREE.Line(tGeom, lineMat));

        // Bottom edge
        const bVerts = new Float32Array([p0[0], 0, p0[1], p1[0], 0, p1[1]]);
        const bGeom = new THREE.BufferGeometry();
        bGeom.setAttribute('position', new THREE.BufferAttribute(bVerts, 3));
        group.add(new THREE.Line(bGeom, lineMat));
    }
    return group;
}

// Categories that mount to the ceiling instead of sitting on the floor.
// Their vertical position is (ceiling - height/2) rather than (height/2).
const CEILING_MOUNTED = new Set([
    "lamp", "chandelier", "pendant", "pendant_light", "ceiling_lamp",
    "ceiling_light", "ceiling_fan", "light_fixture", "fan",
]);

function isCeilingMounted(f) {
    const cat = (f.category || "").toLowerCase().replace(/\s+/g, "_");
    return CEILING_MOUNTED.has(cat);
}

function verticalCenterFor(f) {
    if (isCeilingMounted(f)) {
        const sc = state.currentScene;
        const H = (sc && sc.room && sc.room.height) || 3.0;
        return H - f.height / 2;   // hangs from ceiling
    }
    return f.height / 2;           // sits on floor
}

function buildFurnitureBox(f) {
    const geom = new THREE.BoxGeometry(f.width, f.height, f.depth);
    const color = MATERIAL_COLORS[f.material] || MATERIAL_COLORS.wood;
    const mat = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.7,
        transparent: true,
        opacity: 0.85,
    });
    const mesh = new THREE.Mesh(geom, mat);
    // Position: data (x,y) -> three (x, z). Ceiling-mounted items hang
    // from the ceiling; floor-mounted items sit on the floor.
    const theta = (f.theta || 0) * Math.PI / 180;
    mesh.position.set(f.x, verticalCenterFor(f), f.y);
    mesh.rotation.y = -theta;
    mesh.userData = { category: f.category, id: f.id };
    return mesh;
}

async function loadFurnitureGLB(modelId) {
    // Check cache
    if (state.glbCache[modelId]) {
        return state.glbCache[modelId].clone();
    }

    return new Promise((resolve) => {
        gltfLoader.load(
            `/api/catalog/model/${modelId}/glb`,
            (gltf) => {
                const obj = gltf.scene;
                state.glbCache[modelId] = obj;
                resolve(obj.clone());
            },
            undefined,
            () => resolve(null) // 404 or error -> fallback
        );
    });
}

async function addFurnitureToScene(f) {
    let obj = null;
    if (f.model_id && f.model_id.length > 8) {
        // Only try GLB for real model IDs (not short UUIDs from generic)
        obj = await loadFurnitureGLB(f.model_id);
    }

    if (obj) {
        // Scale GLB to match furniture dimensions
        // GLB is already Y-up (GLTF native), so Y=height
        const box = new THREE.Box3().setFromObject(obj);
        const size = box.getSize(new THREE.Vector3());
        const sx = f.width / (size.x || 1);
        const sy = f.height / (size.y || 1);
        const sz = f.depth / (size.z || 1);
        obj.scale.set(sx, sy, sz);

        // We keep the real 3D-FUTURE PBR textures — the schematic override
        // used to strip them here, which made every piece look like one flat
        // color of wood/fabric. Textured furniture over textured walls looks
        // right in both realistic and schematic modes.

        // Recompute bounding box after scale to center properly
        const box2 = new THREE.Box3().setFromObject(obj);
        const center = box2.getCenter(new THREE.Vector3());
        const minY = box2.min.y;

        // Wrap in a group for consistent transforms
        const wrapper = new THREE.Group();
        wrapper.add(obj);
        // Ceiling-mounted items hang from the ceiling; floor-mounted sit
        // on the floor at y=0.
        if (isCeilingMounted(f)) {
            const sc = state.currentScene;
            const H = (sc && sc.room && sc.room.height) || 3.0;
            wrapper.position.set(f.x, H - f.height, f.y);
        } else {
            wrapper.position.set(f.x, 0, f.y);
        }
        obj.position.set(-center.x, -minY, -center.z);
        const theta = (f.theta || 0) * Math.PI / 180;
        wrapper.rotation.y = -theta;

        wrapper.userData = { category: f.category, id: f.id };
        return wrapper;
    }

    // Fallback: colored box
    return buildFurnitureBox(f);
}

function highlightFurniture(idx) {
    const room = state.currentScene && state.currentScene.room;
    if (!room || !room.furniture[idx]) return;
    const targetId = room.furniture[idx].id;

    layerGroups.furniture.traverse(child => {
        if (child.isMesh) {
            // Walk up to find userData.id
            let obj = child;
            while (obj && !obj.userData.id && obj.parent && obj.parent !== layerGroups.furniture) {
                obj = obj.parent;
            }
            if (obj && obj.userData.id === targetId) {
                child.material = child.material.clone();
                child.material.emissive = new THREE.Color(0x4CD964);
                child.material.emissiveIntensity = 0.4;
            }
        }
    });
}

// ─── Furniture handle creation & selection ────────────

function findFurnitureMeshByIdx(idx) {
    const room = state.currentScene && state.currentScene.room;
    if (!room || !room.furniture[idx]) return null;
    const targetId = room.furniture[idx].id;
    let found = null;
    layerGroups.furniture.children.forEach(child => {
        if (child.userData && child.userData.id === targetId) found = child;
    });
    return found;
}

function removeHandles() {
    while (handleGroup.children.length > 0) {
        const child = handleGroup.children[0];
        handleGroup.remove(child);
        child.traverse(c => {
            if (c.geometry) c.geometry.dispose();
            if (c.material) {
                if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
                else c.material.dispose();
            }
        });
    }
    handleGroup.position.set(0, 0, 0);
    handleGroup.rotation.set(0, 0, 0);
}

function createRotationHandle(furn) {
    const group = new THREE.Group();
    group.userData.handleType = 'rotate';

    const diag = Math.sqrt(furn.width * furn.width + furn.depth * furn.depth) / 2;
    const ringRadius = diag + 0.15;

    // Visible yellow ring
    const torusGeom = new THREE.TorusGeometry(ringRadius, 0.025, 8, 48);
    torusGeom.rotateX(Math.PI / 2); // lay flat on XZ
    const torusMat = new THREE.MeshBasicMaterial({
        color: 0xFFD600, transparent: true, opacity: 0.7,
        depthTest: false,
    });
    const torus = new THREE.Mesh(torusGeom, torusMat);
    torus.position.y = 0.01;
    torus.renderOrder = 100;
    torus.userData.handleType = 'rotate';
    group.add(torus);

    // Invisible thicker torus for raycasting (easier to grab)
    const hitGeom = new THREE.TorusGeometry(ringRadius, 0.12, 8, 48);
    hitGeom.rotateX(Math.PI / 2);
    const hitMat = new THREE.MeshBasicMaterial({
        visible: false,
    });
    const hitTorus = new THREE.Mesh(hitGeom, hitMat);
    hitTorus.position.y = 0.01;
    hitTorus.renderOrder = 100;
    hitTorus.userData.handleType = 'rotate';
    group.add(hitTorus);

    // Small orientation indicator sphere on the ring
    const sphereGeom = new THREE.SphereGeometry(0.06, 8, 8);
    const sphereMat = new THREE.MeshBasicMaterial({
        color: 0xFFD600, depthTest: false,
    });
    const sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.position.set(ringRadius, 0.01, 0);
    sphere.renderOrder = 103;
    sphere.userData.handleType = 'rotate';
    group.add(sphere);

    return group;
}

function createAndPositionHandles(furn) {
    removeHandles();
    const rotate = createRotationHandle(furn);
    handleGroup.add(rotate);
    handleGroup.position.set(furn.x, 0, furn.y);
    handleGroup.rotation.y = -((furn.theta || 0) * Math.PI / 180);
}

function selectFurniture(idx) {
    const room = state.currentScene && state.currentScene.room;
    if (!room || !room.furniture[idx]) return;

    // If different furniture already selected, deselect first
    if (selectedFurnIdx >= 0 && selectedFurnIdx !== idx) {
        removeHandles();
        drawSceneViewport(); // redraw to clear old highlight
    }

    selectedFurnIdx = idx;
    furnInteractionState = FurnInteraction.SELECTED;
    selectedFurnMeshRef = findFurnitureMeshByIdx(idx);

    const furn = room.furniture[idx];
    highlightFurniture(idx);
    createAndPositionHandles(furn);

    document.getElementById('furniture-move-hint').style.display = 'block';
}

function deselectFurniture() {
    removeHandles();
    selectedFurnIdx = -1;
    selectedFurnMeshRef = null;
    furnInteractionState = FurnInteraction.IDLE;
    controls.enabled = true;
    document.getElementById('furniture-move-hint').style.display = 'none';
    const canvas = document.getElementById('viewport-canvas');
    if (canvas) canvas.style.cursor = 'default';
    drawSceneViewport();
}

// Fixed absolute dBm scale for consistent colors across all scenes
// -30 dBm = excellent (deep red), -120 dBm = no signal (deep blue)
const COVERAGE_DB_MIN = -120;
const COVERAGE_DB_MAX = -30;

function buildCoverageHeatmap(coverageData, w, l, zH) {
    const rows = coverageData.length;
    const cols = coverageData[0].length;

    const range = COVERAGE_DB_MAX - COVERAGE_DB_MIN;

    // Create DataTexture (RGBA)
    const data = new Uint8Array(cols * rows * 4);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const clamped = Math.max(COVERAGE_DB_MIN, Math.min(COVERAGE_DB_MAX, coverageData[r][c]));
            const t = (clamped - COVERAGE_DB_MIN) / range; // 0..1
            const [cr, cg, cb] = coverageColormap(t);
            const idx = (r * cols + c) * 4;
            data[idx] = cr;
            data[idx + 1] = cg;
            data[idx + 2] = cb;
            data[idx + 3] = 180; // alpha
        }
    }

    const tex = new THREE.DataTexture(data, cols, rows, THREE.RGBAFormat);
    tex.flipY = true;  // Align texture Y with scene Y (row 0 = south = Three.js z=0)
    tex.needsUpdate = true;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;

    // PlaneGeometry with DoubleSide — visible from both above and below
    const geom = new THREE.PlaneGeometry(w, l);
    geom.rotateX(-Math.PI / 2);
    geom.translate(w / 2, 0, l / 2);

    const mat = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        opacity: 0.75,
        side: THREE.DoubleSide,
        depthWrite: false,
    });

    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.y = zH; // three.js Y = data Z
    mesh.renderOrder = 1;

    // Update colorbar with fixed scale
    updateColorbar(COVERAGE_DB_MIN, COVERAGE_DB_MAX);

    return mesh;
}

function coverageColormap(t) {
    // WiFi signal strength: blue (weak, t=0) → cyan → green → yellow → red (strong, t=1)
    if (t < 0.25) {
        const s = t / 0.25;
        return [Math.round(0), Math.round(s * 180), Math.round(180 + s * 75)];           // dark blue → cyan
    } else if (t < 0.5) {
        const s = (t - 0.25) / 0.25;
        return [Math.round(s * 50), Math.round(180 + s * 60), Math.round(255 - s * 200)]; // cyan → green
    } else if (t < 0.75) {
        const s = (t - 0.5) / 0.25;
        return [Math.round(50 + s * 205), Math.round(240 - s * 30), Math.round(55 - s * 55)]; // green → yellow
    } else {
        const s = (t - 0.75) / 0.25;
        return [Math.round(255), Math.round(210 - s * 210), Math.round(0)];               // yellow → red
    }
}

function updateColorbar(minVal, maxVal) {
    const overlay = document.getElementById('colorbar-overlay');
    overlay.style.display = 'block';
    const canvas = document.getElementById('colorbar-canvas');
    const ctx = canvas.getContext('2d');
    const h = canvas.height;
    const w = canvas.width;
    for (let y = 0; y < h; y++) {
        const t = 1 - y / h; // top=1, bottom=0
        const [r, g, b] = coverageColormap(t);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(0, y, w, 1);
    }
    const ticks = document.getElementById('cb-ticks');
    ticks.innerHTML = `<span>${COVERAGE_DB_MAX} dB</span><span>${((COVERAGE_DB_MIN + COVERAGE_DB_MAX) / 2).toFixed(0)} dB</span><span>${COVERAGE_DB_MIN} dB</span>`;
}

function buildTxMarker(txX, txY, txZ, sceneScale) {
    const group = new THREE.Group();

    // Scale marker with scene size (0.15 for ~5m rooms, larger for outdoor)
    const markerSize = sceneScale ? Math.max(0.15, sceneScale * 0.012) : 0.15;

    // Diamond marker
    const geom = new THREE.OctahedronGeometry(markerSize, 0);
    const mat = new THREE.MeshStandardMaterial({ color: 0xFF3B30, emissive: 0xFF3B30, emissiveIntensity: 0.5 });
    const diamond = new THREE.Mesh(geom, mat);
    diamond.position.copy(toThree(txX, txY, txZ));
    group.add(diamond);

    // Ring around marker for visibility in large scenes
    if (markerSize > 0.5) {
        const ringGeom = new THREE.RingGeometry(markerSize * 1.5, markerSize * 2, 32);
        ringGeom.rotateX(-Math.PI / 2);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0xFF3B30, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.position.copy(toThree(txX, txY, 0.1));
        group.add(ring);
    }

    // Vertical line from floor to TX
    const lineMat = new THREE.LineBasicMaterial({ color: 0xFF3B30, transparent: true, opacity: 0.5, linewidth: 2 });
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
        toThree(txX, txY, 0),
        toThree(txX, txY, txZ),
    ]);
    const line = new THREE.Line(lineGeom, lineMat);
    group.add(line);

    return group;
}

function raySignalColor(distFromTx, bounces) {
    // Approximate signal strength at this distance using same model as heatmap
    // FSPL + bounce loss → dBm → colormap
    const freq = parseFloat(document.getElementById('frequency').value) || 3.5e9;
    const arraySize = state.antennaConfig.rows * state.antennaConfig.cols;
    const d = Math.max(distFromTx, 0.1);
    const fspl = 20 * Math.log10(d) + 20 * Math.log10(freq) - 147.55;
    const arrayGain = 10 * Math.log10(arraySize);
    const bounceLoss = bounces * 8; // ~8 dB per reflection
    const dbm = 30 + arrayGain - fspl - bounceLoss;
    const t = Math.max(0, Math.min(1, (dbm - COVERAGE_DB_MIN) / (COVERAGE_DB_MAX - COVERAGE_DB_MIN)));
    const rgb = coverageColormap(t);
    return new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
}

function buildRayTraces(rayData, fraction) {
    const group = new THREE.Group();
    if (!rayData || rayData.length === 0) return group;

    for (let ri = 0; ri < rayData.length; ri++) {
        const ray = rayData[ri];
        const pts = ray.points;
        if (pts.length < 2) continue;

        const tx = pts[0];

        // Cumulative segment lengths
        const segLens = [];
        let totalLen = 0;
        for (let s = 1; s < pts.length; s++) {
            const dx = pts[s][0] - pts[s - 1][0];
            const dy = pts[s][1] - pts[s - 1][1];
            const dz = pts[s][2] - pts[s - 1][2];
            segLens.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
            totalLen += segLens[segLens.length - 1];
        }

        const drawLen = totalLen * fraction;
        // Build segments with per-vertex color based on signal strength
        const positions = [];
        const colors = [];
        let cumLen = 0;
        let bounceSoFar = 0;

        positions.push(toThree(pts[0][0], pts[0][1], pts[0][2]));
        const c0 = raySignalColor(0, 0);
        colors.push(c0.r, c0.g, c0.b);

        for (let s = 0; s < segLens.length; s++) {
            if (cumLen + segLens[s] <= drawLen) {
                const p = pts[s + 1];
                positions.push(toThree(p[0], p[1], p[2]));
                cumLen += segLens[s];
                bounceSoFar = s + 1;
                const dist = Math.sqrt((p[0]-tx[0])**2 + (p[1]-tx[1])**2 + (p[2]-tx[2])**2);
                const c = raySignalColor(dist, bounceSoFar);
                colors.push(c.r, c.g, c.b);
            } else {
                const remain = drawLen - cumLen;
                const t = remain / segLens[s];
                const ix = pts[s][0] + t * (pts[s + 1][0] - pts[s][0]);
                const iy = pts[s][1] + t * (pts[s + 1][1] - pts[s][1]);
                const iz = pts[s][2] + t * (pts[s + 1][2] - pts[s][2]);
                positions.push(toThree(ix, iy, iz));
                const dist = Math.sqrt((ix-tx[0])**2 + (iy-tx[1])**2 + (iz-tx[2])**2);
                const c = raySignalColor(dist, bounceSoFar);
                colors.push(c.r, c.g, c.b);
                break;
            }
        }

        if (positions.length >= 2) {
            const geom = new THREE.BufferGeometry().setFromPoints(positions);
            geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            const mat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
            group.add(new THREE.Line(geom, mat));
        }
    }
    return group;
}

// ─── Coverage interpolation ──────────────────────────
function interpolateCoverageGrid(zVal) {
    if (!state.coverageSlices || state.coverageSlices.length === 0) return null;
    const slices = state.coverageSlices;

    let lo = 0, hi = slices.length - 1;
    for (let i = 0; i < slices.length - 1; i++) {
        if (slices[i].z <= zVal && slices[i + 1].z >= zVal) {
            lo = i; hi = i + 1; break;
        }
    }
    if (zVal <= slices[0].z) { lo = 0; hi = 0; }
    if (zVal >= slices[slices.length - 1].z) { lo = slices.length - 1; hi = lo; }

    if (lo === hi) return { grid: slices[lo].grid, z: slices[lo].z };

    const t = (zVal - slices[lo].z) / (slices[hi].z - slices[lo].z);
    const gLo = slices[lo].grid, gHi = slices[hi].grid;
    const grid = gLo.map((row, r) => row.map((val, c) => val + t * (gHi[r][c] - val)));
    return { grid, z: zVal };
}

// ─── Outdoor Scene Builder ───────────────────────────

function buildOutdoorScene(w, l, buildings, roads, trees) {
    clearAllGroups();

    // Ground
    const groundGeom = new THREE.PlaneGeometry(w, l);
    groundGeom.rotateX(-Math.PI / 2);
    groundGeom.translate(w / 2, 0, l / 2);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0x225022,
        roughness: 0.95,
        transparent: true,
        opacity: 0.6,
    });
    layerGroups.floor.add(new THREE.Mesh(groundGeom, groundMat));

    // ─── Roads ───────────────────────────────────────
    // Road centerline → extruded strip along the path
    const roadMat = new THREE.MeshStandardMaterial({
        color: 0x3a3a3a, roughness: 0.95, side: THREE.DoubleSide,
    });
    if (roads && roads.length > 0) {
        for (const r of roads) {
            const cl = r.centerline;
            if (!cl || cl.length < 2) continue;
            const hw = (r.width || 4) / 2; // half-width

            // Build a ribbon mesh: for each segment, create a quad strip
            const positions = [];
            const indices = [];
            for (let i = 0; i < cl.length; i++) {
                const [cx, cy] = cl[i];
                let nx, ny;
                if (i < cl.length - 1) {
                    nx = cl[i + 1][1] - cy;
                    ny = -(cl[i + 1][0] - cx);
                } else {
                    nx = cl[i][1] - cl[i - 1][1];
                    ny = -(cl[i][0] - cl[i - 1][0]);
                }
                const len = Math.sqrt(nx * nx + ny * ny) || 1;
                nx /= len; ny /= len;
                // Left and right points in data space
                // Map to Three.js: data.x → three.x, data.y → three.z, y=0.05 (slightly above ground)
                positions.push(cx + nx * hw, 0.05, cy + ny * hw); // left
                positions.push(cx - nx * hw, 0.05, cy - ny * hw); // right

                if (i < cl.length - 1) {
                    const vi = i * 2;
                    indices.push(vi, vi + 1, vi + 2);
                    indices.push(vi + 1, vi + 3, vi + 2);
                }
            }
            const geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geom.setIndex(indices);
            geom.computeVertexNormals();
            layerGroups.roads.add(new THREE.Mesh(geom, roadMat));
        }
    }

    // ─── Trees ───────────────────────────────────────
    // Trunk: cylinder, Crown: sphere (deciduous) or cone (conifer)
    if (trees && trees.length > 0) {
        const trunkMat = new THREE.MeshStandardMaterial({ color: 0x654321, roughness: 0.9 });
        const crownMat = new THREE.MeshStandardMaterial({ color: 0x228B22, roughness: 0.8 });

        for (const t of trees) {
            const [tx, ty] = t.position;
            const h = t.height || 8;
            const cr = t.crown_radius || 3;
            const ch = t.crown_height || (h * 0.6);
            const tr = t.trunk_radius || 0.15;
            const trunkH = Math.max(0.5, h - ch);

            // Trunk cylinder
            const trunkGeom = new THREE.CylinderGeometry(tr, tr, trunkH, 6);
            const trunk = new THREE.Mesh(trunkGeom, trunkMat);
            trunk.position.set(tx, trunkH / 2, ty);

            // Crown
            let crownGeom;
            if (t.species === 'conifer') {
                crownGeom = new THREE.ConeGeometry(cr, ch, 6);
            } else {
                crownGeom = new THREE.SphereGeometry(cr, 6, 5);
            }
            const crown = new THREE.Mesh(crownGeom, crownMat);
            crown.position.set(tx, trunkH + ch / 2, ty);

            const group = new THREE.Group();
            group.add(trunk);
            group.add(crown);
            layerGroups.trees.add(group);
        }
    }

    // ─── Buildings ────────────────────────────────────
    // Negate Y when building shape, reverse winding for correct normals
    for (const b of buildings) {
        const fp = b.footprint || b.fp;
        const h = b.height || b.h || 12;
        if (!fp || fp.length < 3) continue;

        const rfp = fp.slice().reverse();
        const shape = new THREE.Shape();
        shape.moveTo(rfp[0][0], -rfp[0][1]);
        for (let i = 1; i < rfp.length; i++) shape.lineTo(rfp[i][0], -rfp[i][1]);
        shape.closePath();

        const geom = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
        geom.rotateX(-Math.PI / 2);

        const mat = new THREE.MeshStandardMaterial({
            color: 0xB4B4B4,
            roughness: 0.7,
            transparent: true,
            opacity: 0.75,
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.userData.buildingData = {
            id: b.id, fp: fp, h: h,
            material: b.material || 'concrete',
            name: b.name || null,
        };
        layerGroups.buildings.add(mesh);
    }

    // Camera for overhead
    camera.position.set(w / 2, Math.max(w, l) * 1.2, l / 2 + Math.max(w, l) * 0.5);
    controls.target.set(w / 2, 0, l / 2);
}

// ─── Main Scene Drawer ───────────────────────────────

async function drawSceneViewport() {
    const sc = state.currentScene;
    if (!sc) {
        drawEmpty3dViewport();
        return;
    }

    if (sc.type === 'outdoor') {
        drawOutdoorViewport();
        return;
    }

    saveCamera();

    const room = sc.room || {};
    const w = room.width || parseRoomDims().w;
    const l = room.length || parseRoomDims().l;
    const h = room.height || 3.0;
    const polygon = room.polygon || null;
    const furniture = room.furniture || [];

    clearAllGroups();

    // Floor
    layerGroups.floor.add(buildFloor(w, l, polygon));

    // Walls
    if (state.layers.walls) {
        layerGroups.walls.add(buildWalls(w, l, h, polygon));
    }

    // Furniture (async GLB loading)
    if (state.layers.furniture && furniture.length > 0) {
        const promises = furniture.map(f => addFurnitureToScene(f));
        const objects = await Promise.all(promises);
        for (const obj of objects) {
            if (obj) layerGroups.furniture.add(obj);
        }
    }

    // TX marker
    if (state.layers.tx) {
        const txX = parseFloat(document.getElementById('tx-x').value) || w / 2;
        const txY = parseFloat(document.getElementById('tx-y').value) || l / 2;
        const txH = parseFloat(document.getElementById('tx-height').value) || 2.5;
        layerGroups.tx.add(buildTxMarker(txX, txY, txH, Math.max(w, l)));
    }

    // Coverage heatmap
    if (state.layers.heatmap) {
        let grid = null, zH = parseFloat(document.getElementById('z-height').value) || 1.5;
        if (state.coverageSlices && state.coverageSlices.length > 0) {
            const zVal = parseFloat(document.getElementById('z-slider').value);
            const interp = interpolateCoverageGrid(zVal);
            if (interp) { grid = interp.grid; zH = interp.z; }
        } else if (state.coverageData) {
            grid = state.coverageData.coverage;
        }
        if (grid) {
            layerGroups.heatmap.add(buildCoverageHeatmap(grid, w, l, zH));
        }
    }

    // Rays
    if (state.layers.rays && state.rayData && !state.rayAnimating) {
        layerGroups.rays.add(buildRayTraces(state.rayData, 1.0));
    }

    // Apply layer visibility
    for (const [name, group] of Object.entries(layerGroups)) {
        group.visible = state.layers[name] !== undefined ? state.layers[name] : true;
    }
    layerGroups.floor.visible = true; // floor always visible
    layerGroups.buildings.visible = state.layers.walls;

    restoreCamera();

    // Re-sync furniture selection handles after redraw
    if (furnInteractionState !== FurnInteraction.IDLE && selectedFurnIdx >= 0) {
        const room = state.currentScene && state.currentScene.room;
        if (room && room.furniture[selectedFurnIdx]) {
            selectedFurnMeshRef = findFurnitureMeshByIdx(selectedFurnIdx);
            highlightFurniture(selectedFurnIdx);
            createAndPositionHandles(room.furniture[selectedFurnIdx]);
        } else {
            // Furniture was removed externally
            deselectFurniture();
        }
    }

    clampInputsToScene();
}

function drawEmpty3dViewport() {
    const { w, l } = parseRoomDims();
    const h = 2.7;
    state.currentScene = {
        type: 'indoor',
        room: { width: w, length: l, height: h, polygon: null, furniture: [] },
    };
    camera.position.set(w * 1.2, h * 2, l * 1.5);
    controls.target.set(w / 2, h / 3, l / 2);
    drawSceneViewport();
}

function drawOutdoorViewport() {
    let w, l, buildings, roads, trees;
    if (state.currentScene && state.currentScene.type === 'outdoor' && state.currentScene.buildings) {
        w = state.currentScene.width || 200;
        l = state.currentScene.length || 200;
        buildings = state.currentScene.buildings.map(b => ({
            id: b.id,
            fp: b.footprint || b.fp,
            h: b.height || b.h,
            material: b.material || 'concrete',
            name: b.name || null,
        }));
        roads = state.currentScene.roads || [];
        trees = state.currentScene.trees || [];
    } else {
        const size = document.getElementById('scene-size').value.split('x').map(Number);
        w = size[0] || 200;
        l = size[1] || 200;
        buildings = [
            { id: 'bldg_0', fp: [[20,30],[60,30],[60,60],[20,60]], h: 15 },
            { id: 'bldg_1', fp: [[80,20],[130,20],[130,50],[110,50],[110,70],[80,70]], h: 20 },
            { id: 'bldg_2', fp: [[140,80],[180,80],[180,130],[140,130]], h: 10 },
            { id: 'bldg_3', fp: [[30,100],[70,100],[70,150],[50,150],[50,130],[30,130]], h: 18 },
        ];
        roads = [];
        trees = [];
    }

    buildOutdoorScene(w, l, buildings, roads, trees);

    if (!state.currentScene || state.currentScene.type !== 'outdoor') {
        state.currentScene = { type: 'outdoor', width: w, length: l, buildings, roads, trees };
    }

    // Render heatmap over outdoor scene if coverage data exists
    if (state.layers.heatmap && (state.coverageSlices || state.coverageData)) {
        let grid = null, zH = 1.5;
        if (state.coverageSlices && state.coverageSlices.length > 0) {
            const zVal = parseFloat(document.getElementById('z-slider').value);
            const interp = interpolateCoverageGrid(zVal);
            if (interp) { grid = interp.grid; zH = interp.z; }
        } else if (state.coverageData) {
            grid = state.coverageData.coverage;
        }
        if (grid) {
            layerGroups.heatmap.add(buildCoverageHeatmap(grid, w, l, zH));
        }
    }

    // TX marker for outdoor (scaled for large scenes)
    if (state.layers.tx) {
        const txX = parseFloat(document.getElementById('tx-x').value) || w / 2;
        const txY = parseFloat(document.getElementById('tx-y').value) || l / 2;
        const txH = parseFloat(document.getElementById('tx-height').value) || 10;
        layerGroups.tx.add(buildTxMarker(txX, txY, txH, Math.max(w, l)));
    }

    // Rays (persisted from animation)
    if (state.layers.rays && state.rayData && !state.rayAnimating) {
        layerGroups.rays.add(buildRayTraces(state.rayData, 1.0));
    }

    // Apply layer visibility
    for (const [name, group] of Object.entries(layerGroups)) {
        group.visible = state.layers[name] !== undefined ? state.layers[name] : true;
    }
    layerGroups.floor.visible = true;
    layerGroups.buildings.visible = state.layers.walls;

    clampInputsToScene();
}

// ─── Sidebar Tab Switching ───────────────────────────

function switchSidebarTab(tabName) {
    state.sidebarTab = tabName;
    document.querySelectorAll('.sidebar-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.sidebarTab === tabName);
    });
    document.querySelectorAll('.tab-content').forEach(tc => {
        tc.classList.toggle('active', tc.id === `tab-${tabName}`);
    });
}

// ─── View switching ──────────────────────────────────
function switchView(view) {
    state.view = view;
    // Update the label chip in header
    const label = document.querySelector('#scene-type-label .scene-tab');
    if (label) label.textContent = view === 'outdoor' ? 'Outdoor' : 'Indoor';
    // Show/hide sidebar controls based on view
    document.getElementById('indoor-dims').style.display = view === 'outdoor' ? 'none' : 'grid';
    document.getElementById('outdoor-bbox').style.display = view === 'outdoor' ? 'grid' : 'none';
    document.getElementById('back-btn').style.display = 'none';
    // Hide furniture panel for outdoor, show OSM config instead
    document.getElementById('furniture-panel').style.display = view === 'outdoor' ? 'none' : '';
    document.getElementById('osm-config-panel').style.display = view === 'outdoor' ? '' : 'none';
}

function resetUI() {
    state.coverageData = null;
    state.coverageSlices = null;
    state.rayData = null;
    state.rayAnimating = false;
    state.selectedBuilding = null;
    state._selectedBuildingMesh = null;
    document.getElementById('z-slider-overlay').style.display = 'none';
    document.getElementById('colorbar-overlay').style.display = 'none';
    document.getElementById('building-tooltip').style.display = 'none';
    document.getElementById('peak-signal').innerHTML = '-- <span class="metric-unit">dBm</span>';
    clearGroup(layerGroups.heatmap);
    clearGroup(layerGroups.rays);
    clearGroup(layerGroups.tx);
}

function parseRoomDims() {
    const parts = document.getElementById('room-dims').value.split('x').map(s => parseFloat(s.trim()));
    return { w: parts[0] || 5, l: parts[1] || 4 };
}

function clampInputsToScene() {
    const sc = state.currentScene;
    if (!sc) return;
    const isIndoor = sc.type !== 'outdoor';
    const room = sc.room || {};
    const w = isIndoor ? (room.width || 5) : (sc.width || 200);
    const l = isIndoor ? (room.length || 4) : (sc.length || 200);
    const h = isIndoor ? (room.height || 3.0) : 100;

    // TX position
    const txX = document.getElementById('tx-x');
    const txY = document.getElementById('tx-y');
    const txH = document.getElementById('tx-height');
    txX.min = 0; txX.max = w; txX.step = 0.1;
    txY.min = 0; txY.max = l; txY.step = 0.1;
    txH.min = 0.1; txH.max = h; txH.step = 0.1;
    txX.value = Math.min(Math.max(parseFloat(txX.value) || 0, 0), w).toFixed(1);
    txY.value = Math.min(Math.max(parseFloat(txY.value) || 0, 0), l).toFixed(1);
    txH.value = Math.min(Math.max(parseFloat(txH.value) || 0, 0.1), h).toFixed(1);

    // Coverage Z height
    const zH = document.getElementById('z-height');
    zH.min = 0; zH.max = h; zH.step = 0.1;
    zH.value = Math.min(Math.max(parseFloat(zH.value) || 1.5, 0), h).toFixed(1);

    // Max depth
    const md = document.getElementById('max-depth');
    md.min = 1; md.max = 10; md.step = 1;

    // Cell size
    const cs = document.getElementById('cell-size');
    cs.min = 0.05; cs.max = isIndoor ? 1 : 10; cs.step = 0.05;

    // TX power
    const txP = document.getElementById('tx-power');
    txP.min = -10; txP.max = 50; txP.step = 1;

    // Num rays
    const nr = document.getElementById('num-rays');
    nr.min = 4; nr.max = 200; nr.step = 1;
}

// ─── Init ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initThreeViewport();
    drawEmpty3dViewport();
    loadBER();
    drawEmptyHistogram();
    drawEmptyPDP();
    refreshSceneList();
    loadCatalogCategories();
    initFurnitureSearch();
    updatePlacedFurnitureList();

    document.getElementById('z-slider').addEventListener('input', onZSliderChange);

    // ─── Draggable number inputs ─────────────────────
    initDraggableNumbers();

    // Auto-compute: debounce config changes → recompute coverage (2s delay)
    let _autoComputeTimer = null;
    function scheduleAutoCompute() {
        clearTimeout(_autoComputeTimer);
        _autoComputeTimer = setTimeout(() => {
            if (state.currentScene) computeCoverage();
        }, 2000);
    }
    // Analysis tab inputs that should trigger auto-compute
    for (const id of ['frequency', 'tx-power', 'tx-x', 'tx-y', 'tx-height', 'z-height', 'max-depth', 'cell-size']) {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', scheduleAutoCompute);
    }
    window._scheduleAutoCompute = scheduleAutoCompute;
});

// ─── Draggable Number Inputs ─────────────────────────

function initDraggableNumbers() {
    const draggableIds = ['tx-power', 'tx-x', 'tx-y', 'tx-height', 'z-height', 'max-depth', 'cell-size', 'num-rays', 'ant-rows', 'ant-cols'];
    for (const id of draggableIds) {
        const el = document.getElementById(id);
        if (el) {
            el.setAttribute('data-draggable', 'true');
            setupDraggable(el);
        }
    }
    // Room dims is a text field "W x L" — handle separately
    const roomDims = document.getElementById('room-dims');
    if (roomDims) {
        roomDims.setAttribute('data-draggable', 'true');
        setupDraggableRoomDims(roomDims);
    }
}

function setupDraggable(el) {
    let startX, startVal, pending = false, dragging = false;
    const step = parseFloat(el.step) || (el.id === 'cell-size' ? 0.05 : el.id === 'z-height' ? 0.1 : 0.5);
    const DRAG_THRESHOLD = 4; // px before drag activates

    el.addEventListener('mousedown', (e) => {
        if (document.activeElement === el) return; // already focused for text edit
        startX = e.clientX;
        startVal = parseFloat(el.value) || 0;
        pending = true;
        dragging = false;
    });

    document.addEventListener('mousemove', (e) => {
        if (!pending && !dragging) return;
        const dx = e.clientX - startX;
        if (pending && Math.abs(dx) >= DRAG_THRESHOLD) {
            pending = false;
            dragging = true;
            document.body.style.cursor = 'ew-resize';
            document.body.style.userSelect = 'none';
        }
        if (!dragging) return;
        const newVal = startVal + Math.round(dx / 8) * step;
        const min = el.min !== '' ? parseFloat(el.min) : -Infinity;
        const max = el.max !== '' ? parseFloat(el.max) : Infinity;
        el.value = Math.max(min, Math.min(max, parseFloat(newVal.toFixed(4))));
        el.dispatchEvent(new Event('input'));
    });

    document.addEventListener('mouseup', () => {
        if (pending) {
            // Click without drag — focus the input for text editing
            pending = false;
            el.focus();
            el.select();
            return;
        }
        if (!dragging) return;
        dragging = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        el.dispatchEvent(new Event('change'));
    });
}

function setupDraggableRoomDims(el) {
    let startX, startW, startL, pending = false, dragging = false;
    const DRAG_THRESHOLD = 4;

    el.addEventListener('mousedown', (e) => {
        if (document.activeElement === el) return;
        startX = e.clientX;
        const dims = parseRoomDims();
        startW = dims.w;
        startL = dims.l;
        pending = true;
        dragging = false;
    });

    document.addEventListener('mousemove', (e) => {
        if (!pending && !dragging) return;
        const dx = e.clientX - startX;
        if (pending && Math.abs(dx) >= DRAG_THRESHOLD) {
            pending = false;
            dragging = true;
            document.body.style.cursor = 'ew-resize';
            document.body.style.userSelect = 'none';
        }
        if (!dragging) return;
        const delta = Math.round(dx / 12) * 0.5;
        const w = Math.max(1, startW + delta);
        const l = Math.max(1, startL + delta);
        el.value = `${w.toFixed(1)} x ${l.toFixed(1)}`;
        if (state.currentScene && state.currentScene.room) {
            state.currentScene.room.width = w;
            state.currentScene.room.length = l;
        }
    });

    document.addEventListener('mouseup', () => {
        if (pending) {
            pending = false;
            el.focus();
            el.select();
            return;
        }
        if (!dragging) return;
        dragging = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        el.dispatchEvent(new Event('change'));
    });

    // Also handle manual text edits to room-dims → live redraw
    el.addEventListener('change', () => {
        if (state.currentScene && state.currentScene.room) {
            const dims = parseRoomDims();
            state.currentScene.room.width = dims.w;
            state.currentScene.room.length = dims.l;
            drawSceneViewport();
        }
    });
}

// ─── Catalog Categories ──────────────────────────────

function loadCatalogCategories() {
    fetch('/api/catalog/categories')
    .then(r => r.json())
    .then(data => {
        state.catalogCategories = data.categories;
        if (data.categories && data.categories.length > 0) {
            FURNITURE_CATEGORIES = data.categories.map(c => c.name);
        }
    })
    .catch(() => {});
}

function getCategoryLabel(catName) {
    if (!state.catalogCategories) return catName.replace(/_/g, ' ');
    const cat = state.catalogCategories.find(c => c.name === catName);
    if (cat && cat.count > 0) {
        return `${catName.replace(/_/g, ' ')} (${cat.count})`;
    }
    return catName.replace(/_/g, ' ');
}

// ─── Building selection & drill-in ───────────────────
function selectBuilding(mesh, bldgData) {
    // Deselect previous
    if (state._selectedBuildingMesh && state._selectedBuildingMesh !== mesh) {
        state._selectedBuildingMesh.material.emissive.setHex(0x000000);
        state._selectedBuildingMesh.material.opacity = 0.75;
    }
    state.selectedBuilding = bldgData;
    state._selectedBuildingMesh = mesh;
    mesh.material.emissive.setHex(0x443300);
    mesh.material.opacity = 0.9;
    const bLabel = bldgData.name || bldgData.id;
    setStatus('ready', `Selected ${bLabel} (${bldgData.h.toFixed(0)}m ${bldgData.material})`);
}

function deselectBuilding() {
    if (state._selectedBuildingMesh) {
        state._selectedBuildingMesh.material.emissive.setHex(0x000000);
        state._selectedBuildingMesh.material.opacity = 0.75;
    }
    state.selectedBuilding = null;
    state._selectedBuildingMesh = null;
}

function enterSelectedBuilding() {
    // Disabled — building entry removed
}

function enterBuilding(bldg) {
    state.drillBuilding = bldg;
    // Save outdoor scene so we can return to it
    if (state.currentScene && state.currentScene.type === 'outdoor') {
        state._parentOutdoorScene = state.currentScene;
    }
    document.getElementById('back-btn').style.display = 'block';
    // enter-building-btn removed
    const bLabel = bldg.name || bldg.id;
    setStatus('loading', `Entering ${bLabel}...`);
    showViewportLoader(`Entering ${bLabel}...`);

    // Check if we already have saved furniture for this building
    const savedFurn = (state._parentOutdoorScene &&
        state._parentOutdoorScene.building_furniture &&
        state._parentOutdoorScene.building_furniture[bldg.id]) || null;

    const body = { building: { footprint: bldg.fp, height: bldg.h, material: bldg.material || 'concrete' } };
    if (savedFurn) {
        body.furniture = savedFurn;
    }

    fetch(`/api/building/${bldg.id}/enter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    .then(r => r.json())
    .then(data => {
        if (data.job_id) listenJob(data.job_id, (result) => {
            hideViewportLoader();
            applyIndoorResult(result);
            setStatus('ready', `Inside ${bLabel}`);
        });
    })
    .catch(e => { hideViewportLoader(); setStatus('error', e.message); });
}

function backToOutdoor() {
    // Save current building's furniture to parent scene before leaving
    const bldg = state.drillBuilding;
    const parentScene = state._parentOutdoorScene;
    if (bldg && parentScene && state.currentScene && state.currentScene.room) {
        const furniture = state.currentScene.room.furniture || [];
        if (!parentScene.building_furniture) parentScene.building_furniture = {};
        parentScene.building_furniture[bldg.id] = furniture;

        // Persist to backend if the parent scene has a scene_id
        if (parentScene.scene_id) {
            fetch(`/api/scenes/${parentScene.scene_id}/building-furniture`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ building_id: bldg.id, furniture }),
            }).catch(() => {}); // fire-and-forget
        }
    }

    state.drillBuilding = null;
    state.selectedBuilding = null;
    document.getElementById('back-btn').style.display = 'none';
    // enter-building-btn removed
    switchView('outdoor');
    resetUI();
    // Restore the parent outdoor scene
    if (parentScene) {
        state.currentScene = parentScene;
    }
    drawOutdoorViewport();
}

// ─── Full-scene GLB loader ─────────────────────────────────────────
// When an imported scene (e.g. small-bedroom, uga-arch-thz) has a
// scene.glb, render the whole mesh in the viewport instead of drawing
// generic furniture boxes. The GLB is served by our new
// /api/scenes/<id>/file/<name> route.
let _importedSceneMesh = null;
function clearImportedScene() {
    if (_importedSceneMesh) {
        scene3d.remove(_importedSceneMesh);
        _importedSceneMesh.traverse(o => {
            if (o.geometry) o.geometry.dispose();
            if (o.material) {
                (Array.isArray(o.material) ? o.material : [o.material])
                    .forEach(m => m.dispose && m.dispose());
            }
        });
        _importedSceneMesh = null;
    }
}
async function loadSceneGLB(sceneId) {
    clearImportedScene();
    const url = `/api/scenes/${sceneId}/file/scene.glb`;
    // HEAD to check availability without downloading
    try {
        const head = await fetch(url, { method: 'HEAD' });
        if (!head.ok) return null;
    } catch { return null; }
    // If the load response already gave us a furniture list (e.g. apartment-01
    // with metadata.json), we render per-item textured meshes via
    // drawSceneViewport → addFurnitureToScene. In that case hide the
    // pre-baked GLB's TEXTURED sub-meshes (the furniture pieces) so we don't
    // stack a duplicate on top of them, but keep the ColorVisuals sub-meshes
    // (walls, floor, ceiling, interior partitions) visible.
    const sc = state.currentScene;
    const suppressFurn = !!(sc && sc.room && Array.isArray(sc.room.furniture)
        && sc.room.furniture.length > 0);
    return new Promise((resolve) => {
        gltfLoader.load(
            url,
            (gltf) => {
                const obj = gltf.scene;
                obj.userData._imported = sceneId;
                if (suppressFurn) {
                    obj.traverse(child => {
                        if (child.isMesh && child.material && child.material.map) {
                            // Textured mesh = a furniture piece; hide it so the
                            // per-item textured copy renders alone (and stays
                            // draggable).
                            child.visible = false;
                        }
                    });
                }
                // GLB is Y-up (Three.js native), scene is placed in world XZ frame
                scene3d.add(obj);
                _importedSceneMesh = obj;
                console.log(`[cc] loaded ${sceneId}/scene.glb (${obj.children.length} nodes)`);
                appendChatMsg('system',
                    `Loaded ${sceneId}/scene.glb into the viewport.`);
                resolve(obj);
            },
            undefined,
            (err) => { console.warn('GLB load failed', err); resolve(null); }
        );
    });
}

function applyIndoorResult(result) {
    const room = result.room;
    switchView('indoor');
    resetUI();
    clearImportedScene();  // remove any previously imported scene mesh

    state.currentScene = {
        type: 'indoor',
        room: {
            width: room.width,
            length: room.length,
            height: room.height || 3.0,
            polygon: room.polygon || null,
            furniture: room.furniture || [],
            rooms: room.rooms || result.rooms || [],
        },
        scene_id: result.scene_id,
        xml_path: result.xml_path,
    };

    // If this scene has a scene.glb (imported / classmate's, etc.), render
    // the full GLB in the viewport in addition to (or instead of) the
    // generic furniture boxes.
    if (result.scene_id && Array.isArray(result.files) &&
        result.files.includes('scene.glb')) {
        state.currentScene._hasPrebakedGLB = true;
        loadSceneGLB(result.scene_id);
    }

    // If inside a building, auto-save furniture to parent outdoor scene
    const bldg = state.drillBuilding;
    const parentScene = state._parentOutdoorScene;
    if (bldg && parentScene) {
        const furniture = room.furniture || [];
        if (!parentScene.building_furniture) parentScene.building_furniture = {};
        parentScene.building_furniture[bldg.id] = furniture;
        // Persist to backend so it survives page refresh
        if (parentScene.scene_id) {
            fetch(`/api/scenes/${parentScene.scene_id}/building-furniture`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ building_id: bldg.id, furniture }),
            }).catch(() => {});
        }
    }

    document.getElementById('room-dims').value = `${room.width} x ${room.length}`;
    document.getElementById('tx-x').value = (room.width / 2).toFixed(1);
    document.getElementById('tx-y').value = (room.length / 2).toFixed(1);

    // Set camera for room
    const w = room.width, l = room.length, h = room.height || 3.0;
    camera.position.set(w * 1.2, h * 2, l * 1.5);
    controls.target.set(w / 2, h / 3, l / 2);

    drawSceneViewport();
    updatePlacedFurnitureList();
    // Auto-compute coverage
    setTimeout(() => computeCoverage(), 300);
}

// ─── Scene creation ──────────────────────────────────
function createScene() {
    setStatus('loading', 'Creating scene...');
    showViewportLoader('Creating scene...');

    if (state.view === 'outdoor') {
        createOutdoorScene();
        return;
    }

    const dims = document.getElementById('room-dims').value.split('x').map(s => parseFloat(s.trim()));
    const w = dims[0] || 5, l = dims[1] || 4;
    const polyStr = document.getElementById('polygon-input').value.trim();
    let polygon = null;

    if (polyStr) {
        try {
            polygon = polyStr.match(/\([\d.,\s]+\)/g).map(s => {
                const nums = s.replace(/[()]/g, '').split(',').map(Number);
                return [nums[0], nums[1]];
            });
        } catch (e) {
            setStatus('error', 'Invalid polygon format');
            return;
        }
    }

    const body = { room_width: w, room_length: l };
    if (polygon) body.floor_polygon = polygon;
    // Apply any preferences the user queued via chat actions
    // (set_material, set_room_height) before the scene existed.
    if (state.pendingMaterials) {
        if (state.pendingMaterials.wall)    body.wall_material    = state.pendingMaterials.wall;
        if (state.pendingMaterials.floor)   body.floor_material   = state.pendingMaterials.floor;
        if (state.pendingMaterials.ceiling) body.ceiling_material = state.pendingMaterials.ceiling;
    }
    if (state.pendingRoomHeight) body.room_height = state.pendingRoomHeight;

    fetch('/api/scene/indoor/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    .then(r => r.json())
    .then(data => {
        if (data.job_id) listenJob(data.job_id, (result) => {
            hideViewportLoader();
            applyIndoorResult(result);
            setStatus('ready', 'Scene created');
            refreshSceneList();
        });
    })
    .catch(e => { hideViewportLoader(); setStatus('error', e.message); });
}

function createOutdoorScene() {
    const size = document.getElementById('scene-size').value.split('x').map(Number);
    const w = size[0] || 200, l = size[1] || 200;

    const buildings = [
        { id: 'bldg_0', footprint: [[20,30],[60,30],[60,60],[20,60]], height: 15, material: 'concrete' },
        { id: 'bldg_1', footprint: [[80,20],[130,20],[130,50],[110,50],[110,70],[80,70]], height: 20, material: 'concrete' },
        { id: 'bldg_2', footprint: [[140,80],[180,80],[180,130],[140,130]], height: 10, material: 'concrete' },
        { id: 'bldg_3', footprint: [[30,100],[70,100],[70,150],[50,150],[50,130],[30,130]], height: 18, material: 'brick' },
    ];

    fetch('/api/scene/outdoor/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ width: w, length: l, buildings }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.job_id) listenJob(data.job_id, (result) => {
            switchView('outdoor');
            state.currentScene = {
                type: 'outdoor',
                width: result.width,
                length: result.length,
                buildings: result.buildings || [],
                scene_id: result.scene_id,
            };
            resetUI();
            drawOutdoorViewport();
            setStatus('ready', `Outdoor scene: ${result.num_buildings} buildings`);
            refreshSceneList();
            // Auto-compute coverage
            setTimeout(() => computeCoverage(), 300);
        });
    })
    .catch(e => setStatus('error', e.message));
}

// ─── Fetch OSM Scene ─────────────────────────────────

function fetchOSMScene() {
    const locInput = document.getElementById('osm-location').value.trim();
    const radius = parseInt(document.getElementById('osm-radius').value) || 200;
    const ground = document.getElementById('osm-ground-material').value;

    if (!locInput) {
        setStatus('error', 'Enter a location (lat,lon or place name)');
        return;
    }

    setStatus('loading', `Fetching from OSM...`);
    showViewportLoader(`Downloading from OSM...`);

    // Detect lat,lon vs place name
    const body = { radius, ground_material: ground };
    const latLonMatch = locInput.match(/^([-\d.]+)\s*,\s*([-\d.]+)$/);
    if (latLonMatch) {
        body.lat = parseFloat(latLonMatch[1]);
        body.lon = parseFloat(latLonMatch[2]);
    } else {
        body.name = locInput;
    }

    fetch('/api/scene/osm/fetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    .then(r => r.json())
    .then(data => {
        if (data.job_id) {
            listenJob(data.job_id, (result) => {
                state.currentScene = {
                    type: 'outdoor',
                    width: result.width,
                    length: result.length,
                    buildings: result.buildings || [],
                    roads: result.roads || [],
                    trees: result.trees || [],
                    scene_id: result.scene_id,
                };
                resetUI();
                switchView('outdoor');
                drawOutdoorViewport();
                hideViewportLoader();
                // Auto-set TX position
                document.getElementById('tx-x').value = (result.width / 2).toFixed(1);
                document.getElementById('tx-y').value = (result.length / 2).toFixed(1);
                document.getElementById('frequency').value = '3.5e9';
                setStatus('ready', `OSM: ${result.num_buildings} buildings loaded`);
                refreshSceneList();
                setTimeout(() => computeCoverage(), 300);
            });
        } else {
            hideViewportLoader();
            setStatus('error', data.error || 'OSM fetch failed');
        }
    })
    .catch(e => {
        hideViewportLoader();
        setStatus('error', e.message);
    });
}

// ─── Scene Browser ───────────────────────────────────

function refreshSceneList() {
    // Returns the promise so callers can await the dropdown being repopulated.
    return fetch('/api/scenes/list')
    .then(r => r.json())
    .then(scenes => {
        const sel = document.getElementById('scene-select');
        // Prefer current scene id, fallback to previous dropdown value
        const activeId = (state.currentScene && state.currentScene.scene_id) || sel.value;
        sel.innerHTML = '<option value="">-- Select scene --</option>';
        for (const s of scenes) {
            const opt = document.createElement('option');
            opt.value = s.id;
            const display = (s.name || s.id).replace(/_/g, ' ');
            opt.textContent = display;
            sel.appendChild(opt);
        }
        if (activeId) sel.value = activeId;
    })
    .catch(() => {});
}

function loadSelectedScene() {
    const sceneId = document.getElementById('scene-select').value;
    if (!sceneId) return;

    const displayName = sceneId.replace(/_/g, ' ');
    setStatus('loading', `Loading ${displayName}...`);
    showViewportLoader(`Loading ${displayName}...`);

    // Returns the promise so action handlers can await the scene being in.
    return fetch(`/api/scenes/${sceneId}/load`)
    .then(r => r.json())
    .then(data => {
        hideViewportLoader();
        if (data.error) {
            setStatus('error', data.error);
            return;
        }

        resetUI();

        if (data.type === 'outdoor') {
            switchView('outdoor');
            const nRoads = (data.roads || []).length;
            const nTrees = (data.trees || []).length;
            state.currentScene = {
                type: 'outdoor',
                width: data.width,
                length: data.length,
                buildings: data.buildings || [],
                roads: data.roads || [],
                trees: data.trees || [],
                scene_id: sceneId,
                building_furniture: data.building_furniture || {},
            };
            // Auto-set outdoor TX defaults
            document.getElementById('tx-x').value = (data.width / 2).toFixed(1);
            document.getElementById('tx-y').value = (data.length / 2).toFixed(1);
            document.getElementById('tx-height').value = '10';
            document.getElementById('frequency').value = '3.5e9';
            drawOutdoorViewport();
            setStatus('ready', `Outdoor: ${(data.buildings || []).length} buildings, ${nRoads} roads, ${nTrees} trees`);
            // Auto-compute
            setTimeout(() => computeCoverage(), 300);
            return;
        }

        // Indoor
        switchView('indoor');
        clearImportedScene();  // clear any previously imported mesh
        state.currentScene = {
            type: 'indoor',
            room: {
                width: data.width,
                length: data.length,
                height: data.height,
                polygon: data.polygon || null,
                furniture: data.furniture || [],
                rooms: data.rooms || [],
            },
            scene_id: sceneId,
        };
        // If this scene has a scene.glb (imported apartment, classmate's
        // small-bedroom, uga-arch-thz, etc.), render the full GLB in the
        // Three.js viewport in addition to the room wireframe.
        if (Array.isArray(data.files) && data.files.includes('scene.glb')) {
            state.currentScene._hasPrebakedGLB = true;
            loadSceneGLB(sceneId);
        }

        document.getElementById('room-dims').value = `${data.width} x ${data.length}`;
        document.getElementById('tx-x').value = (data.width / 2).toFixed(1);
        document.getElementById('tx-y').value = (data.length / 2).toFixed(1);

        const w = data.width, l = data.length, h = data.height || 3.0;
        camera.position.set(w * 1.2, h * 2, l * 1.5);
        controls.target.set(w / 2, h / 3, l / 2);

        drawSceneViewport();
        updatePlacedFurnitureList();
        setStatus('ready', `Loaded ${sceneId} (${data.furniture.length} items)`);
        // Auto-compute
        setTimeout(() => computeCoverage(), 300);
    })
    .catch(e => setStatus('error', e.message));
}

// ─── Furniture Panel ─────────────────────────────────

function initFurnitureSearch() {
    const input = document.getElementById('furniture-search-input');
    const resultsDiv = document.getElementById('furniture-search-results');
    let debounceTimer = null;

    input.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const q = input.value.trim();
        if (q.length === 0) {
            resultsDiv.classList.remove('open');
            return;
        }
        debounceTimer = setTimeout(() => {
            fetch(`/api/catalog/search?q=${encodeURIComponent(q)}&limit=20`)
            .then(r => r.json())
            .then(data => {
                const cats = data.categories || [];
                if (cats.length === 0) {
                    resultsDiv.innerHTML = '<div class="search-result-item" style="color:var(--text-tertiary)">No results</div>';
                } else {
                    resultsDiv.innerHTML = cats.map(c =>
                        `<div class="search-result-item" data-cat="${c.name}" onclick="event.stopPropagation(); window._showCategoryVariants('${c.name}')">
                            <span>${getFurnitureIcon(c.name)} ${c.name.replace(/_/g, ' ')}</span>
                            <span class="count">${c.count > 0 ? c.count + ' variants &#9654;' : 'generic'}</span>
                        </div>`
                    ).join('');
                }
                resultsDiv.classList.add('open');
            })
            .catch(() => { resultsDiv.classList.remove('open'); });
        }, 200);
    });

    input.addEventListener('focus', () => {
        if (input.value.trim().length > 0 && resultsDiv.children.length > 0) {
            resultsDiv.classList.add('open');
        }
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.furniture-search')) {
            resultsDiv.classList.remove('open');
        }
    });
}

function showCategoryVariants(catName) {
    const resultsDiv = document.getElementById('furniture-search-results');
    resultsDiv.innerHTML = '<div class="search-result-item" style="color:var(--text-tertiary)">Loading variants...</div>';

    fetch(`/api/catalog/category/${encodeURIComponent(catName)}/models?limit=30`)
    .then(r => r.json())
    .then(data => {
        const models = data.models || [];
        let html = `<div class="variant-back" onclick="window._backToCategories()">&#8592; Back &middot; ${catName.replace(/_/g, ' ')} (${data.total} total)</div>`;
        html += '<div class="variant-list">';

        if (models.length === 0) {
            html += '<div class="search-result-item" style="color:var(--text-tertiary)">No models found</div>';
        } else {
            html += `<div class="variant-item" onclick="event.stopPropagation(); window._addFurnitureFromSearch('${catName}')">
                <div style="width:36px;height:36px;background:var(--bg-surface);border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;font-size:18px">${getFurnitureIcon(catName)}</div>
                <div class="variant-info">
                    <div class="variant-name">Random ${catName.replace(/_/g, ' ')}</div>
                    <div class="variant-meta">Auto-select from catalog</div>
                </div>
            </div>`;

            for (const m of models) {
                const label = [m.style, catName.replace(/_/g, ' ')].filter(Boolean).join(' ');
                const meta = [m.material, m.theme].filter(Boolean).join(' / ');
                html += `<div class="variant-item" onclick="event.stopPropagation(); window._addSpecificModel('${m.model_id}', '${catName}')">
                    <img src="/api/catalog/model/${m.model_id}/image" loading="lazy" alt="">
                    <div class="variant-info">
                        <div class="variant-name">${label}</div>
                        <div class="variant-meta">${meta || 'No metadata'}</div>
                    </div>
                </div>`;
            }
        }
        html += '</div>';
        resultsDiv.innerHTML = html;
        resultsDiv.classList.add('open');
    })
    .catch(() => {
        resultsDiv.innerHTML = '<div class="search-result-item" style="color:var(--sim-heat-1)">Failed to load variants</div>';
        resultsDiv.classList.add('open');
    });
}

function backToCategories() {
    const input = document.getElementById('furniture-search-input');
    // Re-trigger the search with current query
    input.dispatchEvent(new Event('input'));
}

function addSpecificModel(modelId, category) {
    document.getElementById('furniture-search-results').classList.remove('open');
    document.getElementById('furniture-search-input').value = '';
    addFurnitureDirectly(category, modelId);
}

function addFurnitureFromSearch(catName) {
    document.getElementById('furniture-search-results').classList.remove('open');
    document.getElementById('furniture-search-input').value = '';
    addFurnitureDirectly(catName, null);
}

function addFurnitureDirectly(category, modelId) {
    const sc = state.currentScene;
    const room = sc && sc.room;

    // Get room dimensions from current scene or input fields
    let w, l;
    if (room) {
        w = room.width;
        l = room.length;
    } else {
        ({ w, l } = parseRoomDims());
    }

    const newItem = { category, quantity: 1 };
    if (modelId) newItem.model_id = modelId;

    const body = { room_width: w, room_length: l, furniture: [newItem] };

    // Always preserve existing furniture
    if (room && room.furniture && room.furniture.length > 0) {
        body.existing_furniture = room.furniture;
    }

    // Carry over polygon
    if (room && room.polygon) {
        body.floor_polygon = room.polygon;
    } else {
        const polyStr = document.getElementById('polygon-input').value.trim();
        if (polyStr) {
            try {
                body.floor_polygon = polyStr.match(/\([\d.,\s]+\)/g).map(s => {
                    const nums = s.replace(/[()]/g, '').split(',').map(Number);
                    return [nums[0], nums[1]];
                });
            } catch (e) { /* ignore */ }
        }
    }

    pushUndo();
    setStatus('loading', `Adding ${category.replace(/_/g, ' ')}...`);
    showViewportLoader('Placing furniture...');

    fetch('/api/scene/indoor/create-with-furniture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    .then(r => r.json())
    .then(data => {
        if (data.job_id) listenJob(data.job_id, (result) => {
            hideViewportLoader();
            applyIndoorResult(result);
            setStatus('ready', `Placed ${result.room.num_furniture} items`);
            refreshSceneList();
        });
    })
    .catch(e => {
        hideViewportLoader();
        setStatus('error', e.message);
    });
}


function updatePlacedFurnitureList() {
    const list = document.getElementById('placed-furniture-list');
    const countEl = document.getElementById('placed-count');
    const furniture = (state.currentScene && state.currentScene.room)
        ? (state.currentScene.room.furniture || []) : [];

    countEl.textContent = furniture.length;
    if (furniture.length === 0) {
        list.innerHTML = '<div style="color:var(--text-tertiary);font-size:11px;padding:4px">No furniture placed</div>';
        return;
    }

    list.innerHTML = furniture.map((f, idx) => {
        const tooltipRows = [
            ['Category', f.category],
            f.super_category ? ['Super Category', f.super_category] : null,
            f.style ? ['Style', f.style] : null,
            f.theme ? ['Theme', f.theme] : null,
            f.model_material ? ['Material (3DF)', f.model_material] : null,
            ['RF Material', f.material],
            ['Dimensions', `${f.width.toFixed(3)} x ${f.depth.toFixed(3)} x ${f.height.toFixed(3)} m`],
            ['Position', `(${f.x.toFixed(2)}, ${f.y.toFixed(2)})`],
            ['Rotation', `${(f.theta || 0).toFixed(1)}\u00B0`],
            f.model_id ? ['Model ID', f.model_id.substring(0, 8) + '...'] : null,
        ].filter(Boolean);

        const tooltipHtml = tooltipRows.map(([k, v]) =>
            `<div class="pf-tooltip-row"><span class="pf-tooltip-key">${k}</span><span class="pf-tooltip-val">${v}</span></div>`
        ).join('');

        const hasImage = f.model_id && f.model_id.length > 8;
        const iconHtml = hasImage
            ? `<img class="pf-thumb" src="/api/catalog/model/${f.model_id}/image" loading="lazy" alt="">`
            : `<div class="pf-icon" style="background:${MATERIAL_COLORS_HEX[f.material] || MATERIAL_COLORS_HEX.wood}22">${getFurnitureIcon(f.category)}</div>`;

        return `<div class="placed-furniture-item" data-idx="${idx}">
            ${iconHtml}
            <div class="pf-info">
                <div class="pf-name">${[f.style, f.category.replace(/_/g, ' ')].filter(Boolean).join(' ')}${f.model_material ? ' [' + f.model_material + ']' : ''}</div>
                <div class="pf-dims">${f.width.toFixed(2)} x ${f.depth.toFixed(2)} x ${f.height.toFixed(2)}m &middot; (${f.x.toFixed(1)}, ${f.y.toFixed(1)})</div>
            </div>
            <div class="pf-tooltip">${tooltipHtml}</div>
        </div>`;
    }).join('');
}


// ─── Coverage compute (multi-Z) ─────────────────────
function computeCoverage() {
    // Cancel any in-progress coverage computation
    if (state.activeEventSource) {
        state.activeEventSource.close();
        state.activeEventSource = null;
        showProgress(false);
    }
    setStatus('loading', 'Computing coverage...');
    // No viewport loader — compute runs in background, progress shown in status indicator
    const btn = document.getElementById('btn-compute');
    btn.disabled = true;

    let freq = parseFloat(document.getElementById('frequency').value);
    if (!isFinite(freq) || freq <= 0) freq = 5e9;   // guard: no NaN → server crash
    const sc = state.currentScene;
    const isOutdoor = sc && sc.type === 'outdoor';

    // Use scene dimensions for outdoor, room dims input for indoor
    let w, l;
    if (isOutdoor) {
        w = sc.width || 200;
        l = sc.length || 200;
    } else {
        const dims = parseRoomDims();
        w = dims.w;
        l = dims.l;
    }

    const txX = parseFloat(document.getElementById('tx-x').value) || w / 2;
    const txY = parseFloat(document.getElementById('tx-y').value) || l / 2;
    let resolution = parseFloat(document.getElementById('cell-size').value) || 0.2;
    const arraySize = state.antennaConfig.rows * state.antennaConfig.cols;

    // For large outdoor areas, auto-increase resolution to keep grid manageable
    // Max ~250x250 grid = 62500 cells
    const maxCells = 250;
    const minRes = Math.max(w / maxCells, l / maxCells);
    if (resolution < minRes) resolution = Math.ceil(minRes * 10) / 10;

    const furniture = (sc && sc.room)
        ? (sc.room.furniture || [])
        : [];

    // For outdoor scenes, send building footprints as obstacles
    const buildings = (isOutdoor && sc.buildings)
        ? sc.buildings.map(b => ({
            footprint: b.footprint || b.fp,
            height: b.height || b.h || 12,
            material: b.material || 'concrete',
        }))
        : [];

    const zMax = isOutdoor ? 30 : ((sc && sc.room && sc.room.height) || 3.0);
    const zMin = isOutdoor ? 1.5 : 0.0;
    const zSteps = isOutdoor ? 5 : 15;

    fetch('/api/coverage/thz/multi-z', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            room_width: w, room_length: l,
            frequency: freq,
            tx_x: txX, tx_y: txY, tx_z: parseFloat(document.getElementById('tx-height').value) || (isOutdoor ? 10 : 2.5),
            resolution: resolution,
            array_elements: arraySize,
            furniture: furniture,
            buildings: buildings,
            z_min: zMin,
            z_max: zMax,
            z_steps: zSteps,
            scene_id: (sc && sc.scene_id) || null,
            antenna_pattern: state.antennaConfig.pattern,
            antenna_polarization: state.antennaConfig.polarization,
            antenna_rows: state.antennaConfig.rows,
            antenna_cols: state.antennaConfig.cols,
            antenna_azimuth: state.antennaConfig.azimuth,
            antenna_elevation: state.antennaConfig.elevation,
        }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.job_id) {
            const es = listenJob(data.job_id, (result) => {
                state.coverageSlices = result.slices;
                state.coverageData = null;

                const slider = document.getElementById('z-slider');
                const zVals = result.z_values || result.slices.map(s => s.z);
                slider.min = zVals[0];
                slider.max = zVals[zVals.length - 1];
                slider.step = 0.01;
                slider.value = zVals[Math.floor(zVals.length / 2)];
                document.getElementById('z-slider-overlay').style.display = 'flex';
                updateZTicks(zVals[0], zVals[zVals.length - 1]);
                onZSliderChange();

                drawSceneViewport();

                const midIdx = Math.floor(result.slices.length / 2);
                const midGrid = result.slices[midIdx].grid;
                drawCoverageHistogram(midGrid);
                updateStats({
                    coverage: midGrid,
                    min_dbm: result.min_dbm,
                    max_dbm: result.max_dbm,
                    mean_dbm: result.mean_dbm,
                }, freq, arraySize);

                // Remember measured stats so the chat agent can quote real
                // before/after numbers (sent as coverage_history).
                {
                    const flat = midGrid.flat();
                    const covPct = flat.filter(v => v > -80).length / flat.length * 100;
                    // Per-room mean RSS (apartment scenes carry room bounds),
                    // so the agent can answer "how much did the bedroom drop".
                    const roomMeans = {};
                    const roomsList = (state.currentScene && state.currentScene.room &&
                                       state.currentScene.room.rooms) || [];
                    if (roomsList.length && midGrid.length) {
                        const W = result.room ? result.room.width : parseRoomDims().w;
                        const L = result.room ? result.room.length : parseRoomDims().l;
                        const ny = midGrid.length, nx = midGrid[0].length;
                        for (const r of roomsList) {
                            let s = 0, n = 0;
                            for (let j = 0; j < ny; j++) {
                                const y = (j + 0.5) * L / ny;
                                if (y < r.y0 || y >= r.y1) continue;
                                for (let i = 0; i < nx; i++) {
                                    const x = (i + 0.5) * W / nx;
                                    if (x < r.x0 || x >= r.x1) continue;
                                    s += midGrid[j][i]; n++;
                                }
                            }
                            if (n) roomMeans[r.id] = +(s / n).toFixed(1);
                        }
                    }
                    state.coverageHistory = state.coverageHistory || [];
                    state.coverageHistory.push({
                        rooms: roomMeans,
                        freq_ghz: +(freq / 1e9).toFixed(2),
                        tx: {
                            x: +(parseFloat(document.getElementById('tx-x').value) || 0).toFixed(2),
                            y: +(parseFloat(document.getElementById('tx-y').value) || 0).toFixed(2),
                            z: +(parseFloat(document.getElementById('tx-height').value) || 0).toFixed(2),
                        },
                        engine: result.engine || 'unknown',
                        mean_dbm: +result.mean_dbm.toFixed(1),
                        min_dbm: +result.min_dbm.toFixed(1),
                        max_dbm: +result.max_dbm.toFixed(1),
                        coverage_pct: +covPct.toFixed(1),
                    });
                    if (state.coverageHistory.length > 8) state.coverageHistory.shift();
                }

                // Backend reports e.g. "Sionna RT (RadioMapSolver)" — match loosely.
                const engineLabel = String(result.engine || '').toLowerCase().includes('sionna') ? 'Sionna RT' : 'THz Model';
                document.getElementById('stat-engine').textContent = engineLabel;
                setStatus('ready', `Coverage computed (${result.slices.length} Z slices, ${engineLabel})`);
                btn.disabled = false;

                document.getElementById('peak-signal').innerHTML =
                    `${result.max_dbm.toFixed(0)} <span class="metric-unit">dBm</span>`;
            }, () => {
                // Cleanup: null out active source and re-enable button
                state.activeEventSource = null;
                btn.disabled = false;
            });
            state.activeEventSource = es;
        }
    })
    .catch(e => {
        setStatus('error', e.message);
        btn.disabled = false;
    });
}

function updateZTicks(zMin, zMax) {
    const container = document.getElementById('z-ticks');
    const majorCount = 5;
    const totalTicks = majorCount * 2 - 1; // 5 major + 4 minor = 9
    let html = '';
    for (let i = 0; i < totalTicks; i++) {
        const val = zMax - (i / (totalTicks - 1)) * (zMax - zMin);
        const isMajor = i % 2 === 0;
        if (isMajor) {
            html += `<div class="z-tick"><span class="z-tick-label">${val.toFixed(1)}</span><span class="z-tick-line"></span></div>`;
        } else {
            html += `<div class="z-tick minor"><span class="z-tick-label"></span><span class="z-tick-line"></span></div>`;
        }
    }
    container.innerHTML = html;
}

function onZSliderChange() {
    if (!state.coverageSlices || state.coverageSlices.length === 0) return;
    const zVal = parseFloat(document.getElementById('z-slider').value);
    document.getElementById('z-slider-value').textContent = zVal.toFixed(2);

    // Rebuild only heatmap layer
    clearGroup(layerGroups.heatmap);
    const interp = interpolateCoverageGrid(zVal);
    if (interp && state.layers.heatmap) {
        const sc = state.currentScene;
        let w, l;
        if (sc && sc.type === 'outdoor') {
            w = sc.width || 200;
            l = sc.length || 200;
        } else {
            const room = sc ? sc.room || {} : {};
            w = room.width || parseRoomDims().w;
            l = room.length || parseRoomDims().l;
        }
        layerGroups.heatmap.add(buildCoverageHeatmap(interp.grid, w, l, interp.z));
    }
}

// ─── Ray Animation ───────────────────────────────────

function generateAndAnimateRays() {
    const sc = state.currentScene;
    if (!sc) {
        setStatus('error', 'Load a scene first');
        return;
    }

    let body;
    if (sc.type === 'outdoor') {
        const w = sc.width || 200, l = sc.length || 200;
        const txX = parseFloat(document.getElementById('tx-x').value) || w / 2;
        const txY = parseFloat(document.getElementById('tx-y').value) || l / 2;
        // Convert building footprints to AABB obstacles for ray tracing
        const buildings = (sc.buildings || []).map(b => {
            const fp = b.footprint || b.fp || [];
            return { footprint: fp, height: b.height || b.h || 12 };
        });
        const txH = parseFloat(document.getElementById('tx-height').value) || 10;
        body = {
            tx_x: txX, tx_y: txY, tx_z: txH,
            room_width: w, room_length: l, room_height: 80,
            num_rays: parseInt(document.getElementById('num-rays').value) || 24,
            max_bounces: 3,
            buildings: buildings,
        };
    } else {
        const room = sc.room || {};
        const txX = parseFloat(document.getElementById('tx-x').value) || room.width / 2;
        const txY = parseFloat(document.getElementById('tx-y').value) || room.length / 2;
        const txH = parseFloat(document.getElementById('tx-height').value) || 2.5;
        body = {
            tx_x: txX, tx_y: txY, tx_z: txH,
            room_width: room.width,
            room_length: room.length,
            room_height: room.height || 3.0,
            num_rays: parseInt(document.getElementById('num-rays').value) || 24,
            max_bounces: 3,
            furniture: room.furniture || [],
        };
    }

    setStatus('loading', 'Generating rays...');
    const btn = document.getElementById('btn-animate-rays');
    btn.disabled = true;

    fetch('/api/rays/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    .then(r => r.json())
    .then(data => {
        state.rayData = data.rays;
        state.layers.rays = true;

        // Update toggle UI
        document.querySelectorAll('.layer-toggle').forEach(el => {
            if (el.querySelector('span').textContent.includes('Ray')) {
                el.classList.add('active');
            }
        });

        animateRays();
        btn.disabled = false;
    })
    .catch(e => {
        setStatus('error', e.message);
        btn.disabled = false;
    });
}

function animateRays() {
    if (!state.rayData || state.rayData.length === 0) return;

    state.rayAnimating = true;
    layerGroups.rays.visible = true;
    setStatus('loading', 'Animating rays...');

    const totalFrames = 40;
    const frameInterval = 50;
    let frame = 0;

    function step() {
        clearGroup(layerGroups.rays);
        const fraction = (frame + 1) / totalFrames;
        layerGroups.rays.add(buildRayTraces(state.rayData, fraction));

        frame++;
        if (frame < totalFrames) {
            setTimeout(step, frameInterval);
        } else {
            state.rayAnimating = false;
            setStatus('ready', `${state.rayData.length} rays rendered`);
        }
    }

    step();
}

// ─── Layer toggling (instant) ────────────────────────

function toggleLayer(el, layer) {
    el.classList.toggle('active');
    state.layers[layer] = el.classList.contains('active');

    // Instant visibility toggle — no full redraw
    if (layerGroups[layer]) {
        layerGroups[layer].visible = state.layers[layer];
    }
    // Buildings follow walls layer
    if (layer === 'walls' && layerGroups.buildings) {
        layerGroups.buildings.visible = state.layers[layer];
    }

    // If toggling heatmap/furniture/tx/rays on and group is empty, do a full redraw
    if (state.layers[layer] && layerGroups[layer] && layerGroups[layer].children.length === 0) {
        drawSceneViewport();
    }
}

// ─── Analysis panels (Plotly, unchanged) ─────────────
function loadBER() {
    fetch('/api/ber/compute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
    })
    .then(r => r.json())
    .then(data => {
        const ht = 'SNR: %{x:.1f} dB<br>BER: %{customdata:.2e}<extra>%{fullData.name}</extra>';
        const traces = [
            { x: data.snr_db, y: data.ber_uncoded.map(v => Math.log10(v)),
              customdata: data.ber_uncoded,
              mode: 'lines', line: { color: '#555', width: 1, dash: 'dot' }, name: 'Uncoded',
              hovertemplate: ht },
            { x: data.snr_db, y: data.ber_ldpc.map(v => Math.log10(v)),
              customdata: data.ber_ldpc,
              mode: 'lines', line: { color: '#E6E0D4', width: 1.5 }, name: 'LDPC',
              hovertemplate: ht },
            { x: data.snr_db, y: data.ber_polar.map(v => Math.log10(v)),
              customdata: data.ber_polar,
              mode: 'lines', line: { color: '#4CD964', width: 1.5 }, name: 'Polar',
              hovertemplate: ht },
        ];
        Plotly.newPlot('ber-chart', traces, {
            ...smallLayout,
            xaxis: { ...smallLayout.xaxis, title: 'SNR (dB)' },
            yaxis: { ...smallLayout.yaxis, title: 'log10(BER)' },
            showlegend: true,
            legend: { font: { size: 8, color: '#9E9E9E' }, x: 0.6, y: 0.95 },
        }, { responsive: true, displayModeBar: false });
    });
}

function drawEmptyHistogram() {
    Plotly.newPlot('coverage-hist', [{
        x: [-90, -80, -70, -60, -50, -40, -30],
        y: [2, 5, 8, 12, 7, 4, 1],
        type: 'bar',
        marker: { color: '#4A4844' },
    }], {
        ...smallLayout,
        xaxis: { ...smallLayout.xaxis, title: 'dBm' },
        yaxis: { ...smallLayout.yaxis, title: 'Count' },
    }, { responsive: true, displayModeBar: false });
}

function drawCoverageHistogram(coverage) {
    const flat = coverage.flat();
    Plotly.react('coverage-hist', [{
        x: flat,
        type: 'histogram',
        nbinsx: 30,
        marker: { color: '#E6E0D4', line: { color: '#050505', width: 0.5 } },
        hovertemplate: '%{x:.1f} dBm<br>Count: %{y}<extra></extra>',
    }], {
        ...smallLayout,
        xaxis: { ...smallLayout.xaxis, title: 'dBm' },
        yaxis: { ...smallLayout.yaxis, title: 'Count' },
    });
}

function drawEmptyPDP() {
    const delays = Array.from({ length: 20 }, (_, i) => i * 5);
    const powers = delays.map(d => -30 - d * 0.8 + Math.random() * 5);
    Plotly.newPlot('pdp-chart', [{
        x: delays, y: powers,
        type: 'bar',
        marker: { color: delays.map(d => d < 15 ? '#E6E0D4' : '#4A4844') },
        hovertemplate: 'Delay: %{x} ns<br>Power: %{y:.1f} dB<extra></extra>',
    }], {
        ...smallLayout,
        xaxis: { ...smallLayout.xaxis, title: 'Delay (ns)' },
        yaxis: { ...smallLayout.yaxis, title: 'Power (dB)' },
    }, { responsive: true, displayModeBar: false });
}

function showSubPanel(panel) {
    document.getElementById('tab-pdp').style.color = panel === 'pdp' ? 'var(--text-primary)' : 'var(--text-tertiary)';
    document.getElementById('tab-ofdm').style.color = panel === 'ofdm' ? 'var(--text-primary)' : 'var(--text-tertiary)';

    if (panel === 'ofdm') {
        fetch('/api/ofdm/grid', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        })
        .then(r => r.json())
        .then(data => {
            Plotly.react('pdp-chart', [{
                z: data.grid,
                type: 'heatmap',
                colorscale: [[0, '#1a1a1a'], [0.5, '#4A4844'], [1, '#E6E0D4']],
                showscale: false,
                hovertemplate: 'Symbol: %{x}<br>Subcarrier: %{y}<br>Power: %{z:.1f} dB<extra></extra>',
            }], {
                ...smallLayout,
                xaxis: { ...smallLayout.xaxis, title: 'Symbol' },
                yaxis: { ...smallLayout.yaxis, title: 'Subcarrier' },
            });
        });
    } else {
        drawEmptyPDP();
    }
}

function updateStats(data, freq, arraySize) {
    const flat = data.coverage.flat();
    const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
    const threshold = -80;
    const covPct = (flat.filter(v => v > threshold).length / flat.length * 100);

    document.getElementById('stat-mean').textContent = `${(-mean).toFixed(1)} dB`;
    document.getElementById('stat-peak').textContent = `${data.max_dbm.toFixed(1)} dBm`;
    document.getElementById('stat-cov').textContent = `${covPct.toFixed(0)}%`;
    document.getElementById('stat-engine').textContent = 'THz Model';
    document.getElementById('stat-freq').textContent = freq >= 1e9 ? `${(freq / 1e9).toFixed(1)} GHz` : `${(freq / 1e6).toFixed(0)} MHz`;

    document.getElementById('stat-array').textContent = `${state.antennaConfig.rows}x${state.antennaConfig.cols}`;
}

// ─── SSE listener ────────────────────────────────────
function listenJob(jobId, onDone, onCleanup) {
    showProgress(true);
    const es = new EventSource(`/api/progress/${jobId}`);
    es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        updateProgress(data.progress, data.message);
        // Show progress in status indicator (top-right)
        if (data.message && data.status === 'running') {
            setStatus('loading', `${data.message} (${data.progress}%)`);
        }
        if (data.status === 'done') {
            es.close();
            showProgress(false);
            if (onCleanup) onCleanup();
            if (onDone) onDone(data.result);
        } else if (data.status === 'error') {
            es.close();
            showProgress(false);
            if (onCleanup) onCleanup();
            setStatus('error', data.message);
        }
    };
    es.onerror = () => {
        es.close();
        showProgress(false);
        if (onCleanup) onCleanup();
        setStatus('error', 'Connection lost');
    };
    return es;
}

// ─── UI helpers ──────────────────────────────────────
function setStatus(type, msg) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    text.textContent = msg;
    dot.style.backgroundColor = type === 'ready' ? '#4CD964' :
                                 type === 'loading' ? '#FF9500' : '#FF3B30';
    dot.style.boxShadow = type === 'ready' ? '0 0 8px rgba(76,217,100,0.4)' :
                           type === 'loading' ? '0 0 8px rgba(255,149,0,0.4)' :
                           '0 0 8px rgba(255,59,48,0.4)';
    dot.classList.toggle('loading', type === 'loading');
}

function showViewportLoader(msg) {
    const el = document.getElementById('viewport-loader');
    document.getElementById('loader-text').textContent = msg || 'Loading...';
    el.classList.add('active');
}

function hideViewportLoader() {
    document.getElementById('viewport-loader').classList.remove('active');
}

function showChatTyping() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-msg typing';
    div.id = 'chat-typing';
    div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function hideChatTyping() {
    const el = document.getElementById('chat-typing');
    if (el) el.remove();
}

function showProgress(show) {
    document.getElementById('progress-wrap').style.display = show ? 'block' : 'none';
    document.getElementById('progress-label').style.display = show ? 'block' : 'none';
    if (!show) {
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-label').textContent = '';
    }
}

function updateProgress(pct, msg) {
    document.getElementById('progress-fill').style.width = `${pct}%`;
    document.getElementById('progress-label').textContent = msg;
}

// ─── Tooltip positioning (fixed, above sidebar) ─────
document.addEventListener('DOMContentLoaded', () => {
    const list = document.getElementById('placed-furniture-list');
    list.addEventListener('mouseover', (e) => {
        const item = e.target.closest('.placed-furniture-item');
        if (!item) return;
        const tooltip = item.querySelector('.pf-tooltip');
        if (!tooltip) return;
        const rect = item.getBoundingClientRect();
        // Position tooltip to the right of sidebar, vertically aligned with item
        tooltip.style.left = (rect.right + 8) + 'px';
        tooltip.style.top = Math.max(8, rect.top) + 'px';
    });
});

// ─── Chat with Claude Haiku ──────────────────────────
const chatMessages = [];

function toggleChat(forceOpen) {
    // With the right-side chat column, messages are always visible and
    // there's no floating history panel to toggle. Keep the function so
    // any existing onfocus/onclick handlers don't throw.
    const history = document.getElementById('chat-history');
    if (!history) return;
    if (forceOpen === true) {
        history.classList.add('open');
    } else {
        history.classList.toggle('open');
    }
}

function appendChatMsg(role, text) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

async function sendChat() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    // Show history panel when sending (no-op when right-side chat column
    // is used — the chat-history popover is not rendered).
    const historyPanel = document.getElementById('chat-history');
    if (historyPanel) historyPanel.classList.add('open');

    input.value = '';
    appendChatMsg('user', msg);
    chatMessages.push({ role: 'user', content: msg });

    const sendBtn = document.getElementById('chat-send');
    sendBtn.disabled = true;
    showChatTyping();

    try {
        const chatPayload = {
            messages: chatMessages,
            scene: state.currentScene,
            // Last few coverage results (newest last) so the agent can quote
            // real before/after numbers instead of guessing or asking.
            coverage_history: state.coverageHistory || [],
            stream: true,   // dispatch async + SSE-stream agent thinking
        };
        if (state.drillBuilding) chatPayload.inside_building = state.drillBuilding;
        if (state.selectedBuilding) chatPayload.selected_building = state.selectedBuilding;

        const kickoff = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(chatPayload),
        });
        const meta = await kickoff.json();

        if (!meta.streaming || !meta.job_id) {
            // Fallback to synchronous response shape.
            hideChatTyping();
            _renderChatFinal(meta);
            sendBtn.disabled = false;
            return;
        }

        // Stream agent events. Each "thinking" / "tool_use" slice gets its
        // own bubble in the chat so the user can see progress instead of
        // staring at a typing dot for 5 minutes.
        await new Promise((resolve) => {
            const es = new EventSource(`/api/progress/${meta.job_id}`);
            es.onmessage = (ev) => {
                let payload;
                try { payload = JSON.parse(ev.data); } catch { return; }
                if (payload.error) {
                    hideChatTyping();
                    appendChatMsg('system', `Error: ${payload.error}`);
                    es.close(); resolve(); return;
                }
                const slices = payload.new_slices || [];
                for (const s of slices) {
                    if (s.kind === 'thinking' && s.text) {
                        appendChatMsg('assistant-thinking', s.text);
                    } else if (s.kind === 'tool_use') {
                        appendChatMsg('tool', `${s.tool}: ${s.summary || ''}`);
                    } else if (s.kind === 'tool_result' && s.ok === false) {
                        appendChatMsg('system', 'tool result: error');
                    }
                }
                if (payload.status === 'done' && payload.result) {
                    hideChatTyping();
                    _renderChatFinal(payload.result);
                    es.close(); resolve();
                } else if (payload.status === 'error') {
                    hideChatTyping();
                    appendChatMsg('system', `Error: ${payload.message || 'agent failed'}`);
                    es.close(); resolve();
                }
            };
            es.onerror = () => {
                hideChatTyping();
                appendChatMsg('system', 'lost connection to agent stream');
                es.close(); resolve();
            };
        });
    } catch (e) {
        hideChatTyping();
        appendChatMsg('system', `Error: ${e.message}`);
    }
    sendBtn.disabled = false;
}


/** Render the final chat reply + panes from a sync /api/chat result OR
 *  the result block of a finished /api/progress SSE message. Shared
 *  between the streaming and fallback paths. */
function _renderChatFinal(data) {
    if (!data || data.error) {
        appendChatMsg('system', `Error: ${(data && data.error) || 'no reply'}`);
        return;
    }
    const reply = data.reply || data.text || '';
    if (reply) {
        appendChatMsg('assistant', reply);
        chatMessages.push({ role: 'assistant', content: reply });
    }
    if (data.actions) {
        // Auto-sort actions by dependency priority BEFORE executing, so
        // Sonnet's occasional ordering slip (e.g. set_material after
        // add_furniture) can't cause silent misapplication.
        const PRIO = {
            'set_material': 0, 'set_room_height': 1, 'set_room_size': 1,
            'add_furniture': 2,
            'set_tx_position': 3, 'set_tx_power': 3, 'set_frequency': 3,
            'configure_antenna': 3, 'set_ap_orientation': 3,
            'move_furniture': 4, 'rotate_furniture': 4, 'remove_furniture': 4,
            'load_scene': 4, 'delete_scene': 4, 'create_apartment': 2,
            'create_outdoor': 4, 'fetch_osm': 4,
            'compute_coverage': 5,
        };
        const sorted = [...data.actions].sort((a, b) =>
            (PRIO[a.type] ?? 99) - (PRIO[b.type] ?? 99));
        // Execute in sorted order and AWAIT each — critical for actions
        // like add_furniture that create the scene, so downstream
        // move_furniture / rotate_furniture see the just-created state.
        (async () => {
            for (const action of sorted) {
                try { await executeChatAction(action); }
                catch (e) { appendChatMsg('system', `Action error: ${e.message}`); }
            }
        })();
    }
    if (data.panes && data.panes.length && window._renderAgentResponse) {
        window._renderAgentResponse({
            text: reply,
            panes: data.panes,
            layout: data.layout || { auto: true },
            files_produced: data.files_produced || [],
        });
    }
}

async function executeChatAction(action) {
    if (action.type === 'add_furniture') {
        appendChatMsg('system', `Adding furniture: ${action.items.map(i => `${i.quantity}x ${i.category}`).join(', ')}`);
        // When inside a building, use the building's polygon room instead of sidebar dims
        const room = state.currentScene && state.currentScene.room;
        const inBuilding = !!state.drillBuilding;
        let w, l;
        if (action.room_width && action.room_length) {
            // AI specified room dimensions (creating a new room)
            w = action.room_width;
            l = action.room_length;
        } else if (inBuilding && room) {
            w = room.width;
            l = room.length;
        } else {
            ({ w, l } = parseRoomDims());
        }
        const body = { room_width: w, room_length: l, furniture: action.items };
        if (inBuilding && room && room.polygon) {
            body.floor_polygon = room.polygon;
        } else {
            const polyStr = document.getElementById('polygon-input').value.trim();
            if (polyStr) {
                try {
                    body.floor_polygon = polyStr.match(/\([\d.,\s]+\)/g).map(s => {
                        const nums = s.replace(/[()]/g, '').split(',').map(Number);
                        return [nums[0], nums[1]];
                    });
                } catch (e) { /* ignore */ }
            }
        }
        // Merge with existing furniture so new items are added, not replaced
        if (room && room.furniture && room.furniture.length > 0) {
            body.existing_furniture = room.furniture;
        }
        // Apply queued material / height preferences from chat actions.
        if (state.pendingMaterials) {
            if (state.pendingMaterials.wall)    body.wall_material    = state.pendingMaterials.wall;
            if (state.pendingMaterials.floor)   body.floor_material   = state.pendingMaterials.floor;
            if (state.pendingMaterials.ceiling) body.ceiling_material = state.pendingMaterials.ceiling;
        }
        if (state.pendingRoomHeight) body.room_height = state.pendingRoomHeight;
        setStatus('loading', 'Optimizing layout...');
        showViewportLoader('Placing furniture...');
        const resp = await fetch('/api/scene/indoor/create-with-furniture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (data.job_id) {
            // Wrap listenJob in a Promise so downstream actions in the same
            // chat batch (e.g. move_furniture on a just-added desk) see the
            // fully-materialised scene.
            await new Promise((resolve) => {
                listenJob(data.job_id, (result) => {
                    hideViewportLoader();
                    applyIndoorResult(result);
                    setStatus('ready', `Placed ${result.room.num_furniture} items`);
                    appendChatMsg('system', `Done! Placed ${result.room.num_furniture} items.`);
                    refreshSceneList();
                    resolve();
                });
            });
        }
    } else if (action.type === 'set_room_size') {
        // Update the sidebar dims field. If a scene already exists, we
        // trigger a rebuild so the 3D viewport reflects the new size
        // right away — otherwise the input change would be silent.
        const w = Number(action.width);
        const l = Number(action.length);
        if (!(w > 0 && l > 0)) {
            appendChatMsg('system', `Invalid room dimensions: ${action.width} x ${action.length}`);
            return;
        }
        document.getElementById('room-dims').value = `${w} x ${l}`;
        const sc = state.currentScene;
        const hasScene = sc && sc.room;
        // If a scene exists AND no subsequent add_furniture is queued (the
        // add_furniture handler rebuilds too), rebuild now so the change
        // is visible immediately. We detect this heuristically by checking
        // if we're the only action being applied in this batch.
        if (hasScene) {
            // Rebuild preserving existing furniture.
            const body = { room_width: w, room_length: l };
            if (sc.room.furniture && sc.room.furniture.length) {
                body.existing_furniture = sc.room.furniture;
            }
            if (state.pendingMaterials) {
                if (state.pendingMaterials.wall)    body.wall_material    = state.pendingMaterials.wall;
                if (state.pendingMaterials.floor)   body.floor_material   = state.pendingMaterials.floor;
                if (state.pendingMaterials.ceiling) body.ceiling_material = state.pendingMaterials.ceiling;
            }
            if (state.pendingRoomHeight) body.room_height = state.pendingRoomHeight;
            try {
                const r = await fetch('/api/scene/indoor/create-with-furniture', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
                const d = await r.json();
                if (d.job_id) {
                    await new Promise((resolve) => {
                        listenJob(d.job_id, (result) => {
                            applyIndoorResult(result);
                            appendChatMsg('system', `Room resized to ${w} × ${l} m`);
                            refreshSceneList();
                            resolve();
                        });
                    });
                }
            } catch (e) {
                appendChatMsg('system', `Resize failed: ${e.message}`);
            }
        } else {
            appendChatMsg('system',
                `Room size set to ${w} × ${l} m (click 'New Scene' or ask me to build to apply).`);
        }
    } else if (action.type === 'set_tx_position') {
        // Move the transmitter (AP) marker in the 3D viewport — clamped
        // to the current room bounds so out-of-range requests don't send
        // the marker off-screen.
        const txx = document.getElementById('tx-x');
        const txy = document.getElementById('tx-y');
        const txh = document.getElementById('tx-height');
        const room = state.currentScene && state.currentScene.room;
        const { w, l } = parseRoomDims();
        const width  = room ? room.width  : w;
        const length = room ? room.length : l;
        const height = room ? (room.height || 3.0) : 3.0;
        const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
        const notes = [];
        if (action.x !== undefined) {
            const raw = Number(action.x);
            const v = clamp(raw, 0, width);
            if (Math.abs(v - raw) > 1e-6) notes.push(`x clamped from ${raw} to ${v}`);
            txx.value = v.toFixed(2);
        }
        if (action.y !== undefined) {
            const raw = Number(action.y);
            const v = clamp(raw, 0, length);
            if (Math.abs(v - raw) > 1e-6) notes.push(`y clamped from ${raw} to ${v}`);
            txy.value = v.toFixed(2);
        }
        if (action.z !== undefined) {
            const raw = Number(action.z);
            const v = clamp(raw, 0.1, height);
            if (Math.abs(v - raw) > 1e-6) notes.push(`z clamped from ${raw} to ${v}`);
            txh.value = v.toFixed(2);
        }
        for (const el of [txx, txy, txh]) {
            el.dispatchEvent(new Event('input', {bubbles:true}));
            el.dispatchEvent(new Event('change', {bubbles:true}));
        }
        if (typeof drawSceneViewport === 'function') drawSceneViewport();
        appendChatMsg('system',
            `AP moved to (${txx.value}, ${txy.value}, ${txh.value}) m` +
            (notes.length ? ` — ${notes.join('; ')}` : ''));
        if (state.currentScene && window._scheduleAutoCompute) window._scheduleAutoCompute();
    } else if (action.type === 'set_tx_power') {
        // Change AP transmit power in dBm.
        const el = document.getElementById('tx-power');
        if (el && action.power_dbm !== undefined) {
            el.value = String(action.power_dbm);
            el.dispatchEvent(new Event('input', {bubbles:true}));
            el.dispatchEvent(new Event('change', {bubbles:true}));
            appendChatMsg('system', `TX power set to ${action.power_dbm} dBm`);
            if (state.currentScene && window._scheduleAutoCompute) window._scheduleAutoCompute();
        }
    } else if (action.type === 'set_frequency') {
        // Change carrier frequency. LLM passes GHz — the select stores Hz
        // strings ("3.5e9", "28e9", ...). Convert to Hz, and if the value
        // isn't already in the dropdown, add an <option> for it so setting
        // the value succeeds (otherwise <select> silently clears to "").
        const el = document.getElementById('frequency');
        if (el && action.ghz !== undefined && action.ghz !== null) {
            const ghz = Number(action.ghz);
            if (!isFinite(ghz) || ghz <= 0) {
                appendChatMsg('system', `Invalid frequency: ${action.ghz}`);
                return;
            }
            const hzStr = String(ghz * 1e9);
            let hasOpt = false;
            for (const opt of el.options) {
                if (Number(opt.value) === ghz * 1e9) { hasOpt = true; break; }
            }
            if (!hasOpt) {
                const o = document.createElement('option');
                o.value = hzStr;
                o.textContent = `${ghz} GHz`;
                el.appendChild(o);
            }
            el.value = hzStr;
            el.dispatchEvent(new Event('input', {bubbles:true}));
            el.dispatchEvent(new Event('change', {bubbles:true}));
            appendChatMsg('system', `Carrier frequency set to ${ghz} GHz`);
            if (state.currentScene && window._scheduleAutoCompute) window._scheduleAutoCompute();
        }
    } else if (action.type === 'move_furniture') {
        // Move a placed piece of furniture by index or category name.
        // Positions are clamped so the furniture's AABB stays inside the
        // room; requests outside get pinned to the wall and reported back.
        const room = state.currentScene && state.currentScene.room;
        if (!room || !room.furniture) {
            appendChatMsg('system', 'No scene loaded — nothing to move.');
            return;
        }
        let idx = -1;
        if (typeof action.index === 'number') idx = action.index;
        else if (action.category) {
            idx = room.furniture.findIndex(f =>
                (f.category || '').toLowerCase() === action.category.toLowerCase());
        }
        if (idx < 0 || idx >= room.furniture.length) {
            appendChatMsg('system', `Cannot find furniture ${action.category || `#${action.index}`}.`);
            return;
        }
        pushUndo();
        const f = room.furniture[idx];
        const hw = (f.width  || 0.5) / 2;
        const hd = (f.depth  || 0.5) / 2;
        const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
        const notes = [];
        if (action.x !== undefined) {
            const raw = Number(action.x);
            const v = clamp(raw, hw, room.width - hw);
            if (Math.abs(v - raw) > 1e-6) notes.push(`x clamped from ${raw} to ${v.toFixed(2)}`);
            f.x = v;
        }
        if (action.y !== undefined) {
            const raw = Number(action.y);
            const v = clamp(raw, hd, room.length - hd);
            if (Math.abs(v - raw) > 1e-6) notes.push(`y clamped from ${raw} to ${v.toFixed(2)}`);
            f.y = v;
        }
        drawSceneViewport();
        updatePlacedFurnitureList();
        saveFurniture();
        appendChatMsg('system',
            `Moved ${f.category} to (${f.x.toFixed(2)}, ${f.y.toFixed(2)}) m` +
            (notes.length ? ` — ${notes.join('; ')}` : ''));
    } else if (action.type === 'rotate_furniture') {
        const room = state.currentScene && state.currentScene.room;
        if (!room || !room.furniture) {
            appendChatMsg('system', 'No scene loaded — nothing to rotate.');
            return;
        }
        let idx = typeof action.index === 'number' ? action.index :
            (action.category ? room.furniture.findIndex(f =>
                (f.category || '').toLowerCase() === action.category.toLowerCase()) : -1);
        if (idx < 0 || idx >= room.furniture.length) {
            appendChatMsg('system', `Cannot find furniture ${action.category || `#${action.index}`}.`);
            return;
        }
        pushUndo();
        const cur = room.furniture[idx].theta || 0;
        if (action.absolute_deg !== undefined) {
            room.furniture[idx].theta = Number(action.absolute_deg) % 360;
        } else if (action.delta_deg !== undefined) {
            room.furniture[idx].theta = (cur + Number(action.delta_deg)) % 360;
        }
        drawSceneViewport();
        saveFurniture();
        appendChatMsg('system',
            `Rotated ${room.furniture[idx].category} to ${room.furniture[idx].theta.toFixed(0)}°`);
    } else if (action.type === 'remove_furniture') {
        const room = state.currentScene && state.currentScene.room;
        if (!room || !room.furniture) return;
        let idx = typeof action.index === 'number' ? action.index :
            (action.category ? room.furniture.findIndex(f =>
                (f.category || '').toLowerCase() === action.category.toLowerCase()) : -1);
        if (idx < 0 || idx >= room.furniture.length) {
            appendChatMsg('system', `Cannot find furniture ${action.category || `#${action.index}`}.`);
            return;
        }
        pushUndo();
        const removed = room.furniture.splice(idx, 1)[0];
        deselectFurniture && deselectFurniture();
        drawSceneViewport();
        updatePlacedFurnitureList();
        saveFurniture();
        appendChatMsg('system', `Removed ${removed.category}`);
    } else if (action.type === 'set_room_height') {
        // If a scene exists, rebuild it with the new ceiling height (so
        // the 3D walls actually grow / shrink). Also update the z-height
        // (coverage layer) input as a sensible sidebar reflection.
        const h = Number(action.height);
        if (!(h > 0 && h < 20)) {
            appendChatMsg('system',
                `Invalid ceiling height ${action.height} (must be 0-20 m).`);
            return;
        }
        const zH = document.getElementById('z-height');
        if (zH) {
            zH.value = String(Math.min(h - 0.1, parseFloat(zH.value) || 1.5));
            zH.dispatchEvent(new Event('input', {bubbles:true}));
        }
        const sc = state.currentScene;
        if (sc && sc.room) {
            const body = {
                room_width: sc.room.width,
                room_length: sc.room.length,
                room_height: h,
            };
            if (sc.room.furniture && sc.room.furniture.length) {
                body.existing_furniture = sc.room.furniture;
            }
            if (state.pendingMaterials) {
                if (state.pendingMaterials.wall)    body.wall_material    = state.pendingMaterials.wall;
                if (state.pendingMaterials.floor)   body.floor_material   = state.pendingMaterials.floor;
                if (state.pendingMaterials.ceiling) body.ceiling_material = state.pendingMaterials.ceiling;
            }
            try {
                const r = await fetch('/api/scene/indoor/create-with-furniture', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body),
                });
                const d = await r.json();
                if (d.job_id) {
                    await new Promise((resolve) => {
                        listenJob(d.job_id, (result) => {
                            applyIndoorResult(result);
                            appendChatMsg('system', `Ceiling height set to ${h} m and scene rebuilt`);
                            refreshSceneList();
                            resolve();
                        });
                    });
                }
            } catch (e) {
                appendChatMsg('system', `Height change failed: ${e.message}`);
            }
        } else {
            // No scene yet — remember height for next scene build.
            state.pendingRoomHeight = h;
            appendChatMsg('system',
                `Ceiling height set to ${h} m (will apply on next scene build).`);
        }
    } else if (action.type === 'set_material') {
        // The indoor sidebar doesn't currently expose per-surface material
        // dropdowns, so we store the preference on `state` and pass it to
        // the next scene-creation call. Show a friendly message so the
        // user knows the change is queued, not silently dropped.
        const surface = (action.surface || 'wall').toLowerCase();
        const mat = action.material;
        if (!mat) {
            appendChatMsg('system', `No material specified.`);
            return;
        }
        // Pre-baked apartment scenes carry a scene.xml that Sionna RT reads
        // directly, so rewrite the surface's ITU material there and
        // recompute. Other scenes keep the old "apply on next build" path.
        const sid = state.currentScene && state.currentScene.scene_id;
        let applied = false;
        if (sid) {
            try {
                const resp = await fetch(`/api/scenes/${sid}/material`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ surface, material: mat }),
                });
                const d = await resp.json();
                if (resp.ok && !d.error) {
                    applied = true;
                    appendChatMsg('system',
                        `${sid}: ${d.surface} material → ${d.material} (${d.shapes_changed} wall panels). Recomputing…`);
                    if (window._scheduleAutoCompute) window._scheduleAutoCompute();
                } else if (resp.status !== 404) {
                    appendChatMsg('system', `Material change failed: ${d.error || resp.status}`);
                    return;
                }
            } catch (e) {
                appendChatMsg('system', `Material change failed: ${e.message}`);
                return;
            }
        }
        if (!applied) {
            state.pendingMaterials = state.pendingMaterials || {};
            state.pendingMaterials[surface] = mat;
            appendChatMsg('system',
                `${surface} material set to ${mat} — will apply the next time you create a scene ('New Scene' or 'Build …' via chat).`);
        }
    } else if (action.type === 'compute_coverage') {
        computeCoverage();
    } else if (action.type === 'load_scene') {
        appendChatMsg('system', `Loading scene: ${action.scene_id}...`);
        document.getElementById('scene-select').value = action.scene_id;
        loadSelectedScene();
    } else if (action.type === 'create_apartment') {
        // Multi-room apartment: server builds walls/partitions/furniture
        // (scene.xml + scene.glb + metadata.json), then we load it like
        // any saved scene. Generation takes ~10-40 s (3D-FUTURE meshes).
        const rooms = Array.isArray(action.rooms) ? action.rooms : [];
        if (!rooms.length) {
            appendChatMsg('system', 'create_apartment: no rooms given.');
            return;
        }
        const desc = rooms.map(r => `${r.type}${r.width && r.depth ? ` ${r.width}×${r.depth}` : ''}`).join(', ');
        appendChatMsg('system', `Building apartment (${rooms.length} rooms: ${desc})… this takes a moment.`);
        showViewportLoader('Generating apartment…');
        try {
            const resp = await fetch('/api/scenes/apartment/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rooms,
                    name: action.name || undefined,
                    height: action.height || undefined,
                    seed: action.seed || undefined,
                }),
            });
            const d = await resp.json();
            hideViewportLoader();
            if (!resp.ok || d.error) {
                appendChatMsg('system', `Apartment generation failed: ${d.error || resp.status}`);
                return;
            }
            appendChatMsg('system',
                `Built ${d.scene_id}: ${d.width}×${d.length}×${d.height} m, ` +
                `${d.rooms.length} rooms, ${d.n_furniture} furniture pieces.`);
            await refreshSceneList();
            document.getElementById('scene-select').value = d.scene_id;
            await loadSelectedScene();
        } catch (e) {
            hideViewportLoader();
            appendChatMsg('system', `Apartment generation failed: ${e.message}`);
        }
    } else if (action.type === 'delete_scene') {
        appendChatMsg('system', `Deleting scene: ${action.scene_id}...`);
        try {
            const resp = await fetch(`/api/scenes/${action.scene_id}/delete`, { method: 'DELETE' });
            const d = await resp.json();
            if (d.deleted) {
                appendChatMsg('system', `Deleted ${action.scene_id}`);
                refreshSceneList();
            } else {
                appendChatMsg('system', `Failed: ${d.error || 'Unknown error'}`);
            }
        } catch (e) { appendChatMsg('system', `Error: ${e.message}`); }
    } else if (action.type === 'set_ap_orientation') {
        // Convenience action to change just azimuth/elevation without
        // touching pattern/rows/cols. Useful for 'rotate the AP east'
        // or 'point the AP toward the corner'.
        const cfg = state.antennaConfig;
        if (action.azimuth !== undefined)   cfg.azimuth   = parseInt(action.azimuth);
        if (action.elevation !== undefined) cfg.elevation = parseInt(action.elevation);
        // Sync sidebar inputs too so hand-drag & re-compute pick up the change
        const azEl = document.getElementById('ant-azimuth');
        const elEl = document.getElementById('ant-elevation');
        if (azEl && action.azimuth   !== undefined) { azEl.value = cfg.azimuth;   azEl.dispatchEvent(new Event('change',{bubbles:true})); }
        if (elEl && action.elevation !== undefined) { elEl.value = cfg.elevation; elEl.dispatchEvent(new Event('change',{bubbles:true})); }
        if (typeof updateAntennaSummary === 'function') updateAntennaSummary();
        drawSceneViewport();
        appendChatMsg('system',
            `AP orientation set to azimuth=${cfg.azimuth}°, elevation=${cfg.elevation}°`);
        // Warn if pattern is iso — orientation has no effect in that case
        if (cfg.pattern === 'iso') {
            appendChatMsg('system',
                `Note: current antenna pattern is "iso" (omnidirectional). ` +
                `Orientation only affects directional patterns — switch to ` +
                `"tr38901" to see the beam rotate. Try: "Use a 4×4 tr38901 antenna."`);
        }
        // Trigger a coverage recompute so the heatmap actually updates
        if (state.currentScene && window._scheduleAutoCompute) window._scheduleAutoCompute();
    } else if (action.type === 'configure_antenna') {
        const cfg = state.antennaConfig;
        if (action.pattern) cfg.pattern = action.pattern;
        if (action.polarization) cfg.polarization = action.polarization;
        if (action.rows) cfg.rows = parseInt(action.rows);
        if (action.cols) cfg.cols = parseInt(action.cols);
        if (action.azimuth !== undefined) cfg.azimuth = parseInt(action.azimuth);
        if (action.elevation !== undefined) cfg.elevation = parseInt(action.elevation);
        updateAntennaSummary();
        appendChatMsg('system', `Antenna configured: ${cfg.rows}x${cfg.cols} ${cfg.pattern}, ${cfg.polarization} pol, az=${cfg.azimuth}°, el=${cfg.elevation}°`);
        // Trigger coverage recompute so pattern/orientation change is visible
        if (state.currentScene && window._scheduleAutoCompute) window._scheduleAutoCompute();
    } else if (action.type === 'create_outdoor') {
        createOutdoorScene();
    } else if (action.type === 'fetch_osm') {
        const locName = action.name || 'location';
        appendChatMsg('system', `Fetching ${locName} from OpenStreetMap...`);
        setStatus('loading', `Downloading OSM: ${locName}...`);
        showViewportLoader(`Downloading ${locName}...`);
        try {
            const resp = await fetch('/api/scene/osm/fetch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(action),
            });
            const d = await resp.json();
            if (d.job_id) {
                listenJob(d.job_id, (result) => {
                    state.currentScene = {
                        type: 'outdoor',
                        width: result.width,
                        length: result.length,
                        buildings: result.buildings || [],
                        roads: result.roads || [],
                        scene_id: result.scene_id,
                    };
                    resetUI();
                    switchView('outdoor');
                    drawOutdoorViewport();
                    hideViewportLoader();
                    setStatus('ready', `OSM: ${result.num_buildings} buildings loaded`);
                    appendChatMsg('system', `Done! Loaded ${result.num_buildings} buildings from OSM.`);
                    refreshSceneList();
                    // Auto-compute coverage
                    setTimeout(() => computeCoverage(), 300);
                });
            } else {
                hideViewportLoader();
                appendChatMsg('system', `Error: ${d.error || 'Unknown error'}`);
                setStatus('error', 'OSM fetch failed');
            }
        } catch (e) {
            hideViewportLoader();
            appendChatMsg('system', `Error: ${e.message}`);
            setStatus('error', 'OSM fetch failed');
        }
    }
}

// ─── Antenna angle preview ───────────────────────────

function drawAntennaPreview() {
    // Top-down view: azimuth
    const topCanvas = document.getElementById('antenna-preview-top');
    if (!topCanvas) return;
    const ctx = topCanvas.getContext('2d');
    const w = topCanvas.width, h = topCanvas.height;
    const cx = w / 2, cy = h / 2;
    const r = Math.min(w, h) / 2 - 8;
    const az = (parseInt(document.getElementById('ant-azimuth').value) || 0) * Math.PI / 180;

    ctx.clearRect(0, 0, w, h);

    // Grid circles
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (const fr of [0.33, 0.66, 1.0]) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * fr, 0, Math.PI * 2);
        ctx.stroke();
    }

    // Compass labels
    ctx.fillStyle = '#666';
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('N', cx, 10);
    ctx.fillText('S', cx, h - 4);
    ctx.fillText('E', w - 4, cy + 3);
    ctx.fillText('W', 6, cy + 3);

    // Antenna coverage wedge (60° beamwidth for tr38901)
    const pattern = document.querySelector('.antenna-pattern-card.active');
    const patName = pattern ? pattern.dataset.pattern : 'tr38901';
    const beamWidth = patName === 'iso' ? Math.PI * 2 : patName === 'dipole' ? Math.PI * 0.8 : Math.PI / 3;

    // Azimuth: 0° = North (+Y), clockwise. Canvas: 0 = right, so offset -PI/2
    const drawAz = az - Math.PI / 2;

    if (patName === 'iso') {
        // Full circle coverage
        ctx.fillStyle = 'rgba(230, 224, 212, 0.15)';
        ctx.beginPath();
        ctx.arc(cx, cy, r * 0.85, 0, Math.PI * 2);
        ctx.fill();
    } else {
        // Wedge
        ctx.fillStyle = 'rgba(230, 224, 212, 0.15)';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r * 0.85, drawAz - beamWidth / 2, drawAz + beamWidth / 2);
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = 'rgba(230, 224, 212, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r * 0.85, drawAz - beamWidth / 2, drawAz + beamWidth / 2);
        ctx.closePath();
        ctx.stroke();
    }

    // Direction arrow
    ctx.strokeStyle = '#E6E0D4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    const arrLen = r * 0.7;
    const ax = cx + Math.cos(drawAz) * arrLen;
    const ay = cy + Math.sin(drawAz) * arrLen;
    ctx.lineTo(ax, ay);
    ctx.stroke();

    // Arrowhead
    const headLen = 6;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(drawAz - 0.4), ay - headLen * Math.sin(drawAz - 0.4));
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(drawAz + 0.4), ay - headLen * Math.sin(drawAz + 0.4));
    ctx.stroke();

    // Center dot
    ctx.fillStyle = '#FF3B30';
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fill();

    // Side view: elevation
    const sideCanvas = document.getElementById('antenna-preview-side');
    if (!sideCanvas) return;
    const sCtx = sideCanvas.getContext('2d');
    const sw = sideCanvas.width, sh = sideCanvas.height;
    const scx = sw / 2, scy = sh / 2;
    const sr = Math.min(sw, sh) / 2 - 8;
    const el = (parseInt(document.getElementById('ant-elevation').value) || 0) * Math.PI / 180;

    sCtx.clearRect(0, 0, sw, sh);

    // Ground line
    sCtx.strokeStyle = '#444';
    sCtx.lineWidth = 1;
    sCtx.beginPath();
    sCtx.moveTo(4, scy);
    sCtx.lineTo(sw - 4, scy);
    sCtx.stroke();

    // Labels
    sCtx.fillStyle = '#666';
    sCtx.font = '8px monospace';
    sCtx.textAlign = 'center';
    sCtx.fillText('UP', scx, 10);
    sCtx.fillText('DOWN', scx, sh - 4);

    // Elevation wedge: 0° = horizontal, positive = down, negative = up
    // In side view: angle from horizontal line, drawing from center
    const elBeam = patName === 'iso' ? Math.PI * 2 : Math.PI / 3;
    const drawEl = el; // positive down

    if (patName === 'iso') {
        sCtx.fillStyle = 'rgba(230, 224, 212, 0.15)';
        sCtx.beginPath();
        sCtx.arc(scx, scy, sr * 0.85, 0, Math.PI * 2);
        sCtx.fill();
    } else {
        sCtx.fillStyle = 'rgba(230, 224, 212, 0.15)';
        sCtx.beginPath();
        sCtx.moveTo(scx, scy);
        sCtx.arc(scx, scy, sr * 0.85, drawEl - elBeam / 2, drawEl + elBeam / 2);
        sCtx.closePath();
        sCtx.fill();

        sCtx.strokeStyle = 'rgba(230, 224, 212, 0.5)';
        sCtx.lineWidth = 1;
        sCtx.beginPath();
        sCtx.moveTo(scx, scy);
        sCtx.arc(scx, scy, sr * 0.85, drawEl - elBeam / 2, drawEl + elBeam / 2);
        sCtx.closePath();
        sCtx.stroke();
    }

    // Direction arrow
    sCtx.strokeStyle = '#E6E0D4';
    sCtx.lineWidth = 2;
    sCtx.beginPath();
    sCtx.moveTo(scx, scy);
    const selLen = sr * 0.7;
    const sex = scx + Math.cos(drawEl) * selLen;
    const sey = scy + Math.sin(drawEl) * selLen;
    sCtx.lineTo(sex, sey);
    sCtx.stroke();

    // Arrowhead
    sCtx.beginPath();
    sCtx.moveTo(sex, sey);
    sCtx.lineTo(sex - headLen * Math.cos(drawEl - 0.4), sey - headLen * Math.sin(drawEl - 0.4));
    sCtx.moveTo(sex, sey);
    sCtx.lineTo(sex - headLen * Math.cos(drawEl + 0.4), sey - headLen * Math.sin(drawEl + 0.4));
    sCtx.stroke();

    // Center dot
    sCtx.fillStyle = '#FF3B30';
    sCtx.beginPath();
    sCtx.arc(scx, scy, 3, 0, Math.PI * 2);
    sCtx.fill();
}

window._drawAntennaPreview = drawAntennaPreview;

// ─── Draggable antenna preview canvases ──────────────
function initAntennaDrag() {
    // Azimuth: drag on top canvas
    const topCanvas = document.getElementById('antenna-preview-top');
    if (topCanvas) {
        let dragging = false;
        topCanvas.addEventListener('pointerdown', (e) => {
            dragging = true;
            topCanvas.setPointerCapture(e.pointerId);
            topCanvas.style.cursor = 'grabbing';
            updateAzFromPointer(e);
        });
        topCanvas.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            updateAzFromPointer(e);
        });
        topCanvas.addEventListener('pointerup', () => {
            dragging = false;
            topCanvas.style.cursor = 'grab';
        });
        function updateAzFromPointer(e) {
            const rect = topCanvas.getBoundingClientRect();
            const cx = rect.width / 2, cy = rect.height / 2;
            const px = e.clientX - rect.left - cx;
            const py = e.clientY - rect.top - cy;
            // atan2 gives angle from +X axis; azimuth 0°=North(up), clockwise
            // Canvas: up = -Y, right = +X. North = -Y direction.
            // angle from north clockwise: atan2(px, -py)
            let deg = Math.atan2(px, -py) * (180 / Math.PI);
            if (deg < 0) deg += 360;
            deg = Math.round(deg);
            document.getElementById('ant-azimuth').value = deg;
            document.getElementById('ant-az-val').textContent = deg + '°';
            drawAntennaPreview();
        }
    }

    // Elevation: drag on side canvas
    const sideCanvas = document.getElementById('antenna-preview-side');
    if (sideCanvas) {
        let dragging = false;
        sideCanvas.addEventListener('pointerdown', (e) => {
            dragging = true;
            sideCanvas.setPointerCapture(e.pointerId);
            sideCanvas.style.cursor = 'grabbing';
            updateElFromPointer(e);
        });
        sideCanvas.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            updateElFromPointer(e);
        });
        sideCanvas.addEventListener('pointerup', () => {
            dragging = false;
            sideCanvas.style.cursor = 'grab';
        });
        function updateElFromPointer(e) {
            const rect = sideCanvas.getBoundingClientRect();
            const cx = rect.width / 2, cy = rect.height / 2;
            const px = e.clientX - rect.left - cx;
            const py = e.clientY - rect.top - cy;
            // Elevation: horizontal=0°, pointing right. atan2(py, px).
            // Positive elevation = downward tilt (below horizon)
            let deg = Math.round(Math.atan2(py, px) * (180 / Math.PI));
            deg = Math.max(-90, Math.min(90, deg));
            document.getElementById('ant-elevation').value = deg;
            document.getElementById('ant-el-val').textContent = deg + '°';
            drawAntennaPreview();
        }
    }
}

// Initialize drag once DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAntennaDrag);
} else {
    initAntennaDrag();
}

// ─── Antenna modal ──────────────────────────────────
function openAntennaModal() {
    const cfg = state.antennaConfig;
    // Sync modal state from config
    document.querySelectorAll('.antenna-pattern-card').forEach(c => c.classList.toggle('active', c.dataset.pattern === cfg.pattern));
    document.querySelectorAll('.pol-btn').forEach(b => b.classList.toggle('active', b.dataset.pol === cfg.polarization));
    document.getElementById('ant-rows').value = cfg.rows;
    document.getElementById('ant-cols').value = cfg.cols;
    document.getElementById('ant-total').textContent = `= ${cfg.rows * cfg.cols} elements`;
    document.getElementById('ant-azimuth').value = cfg.azimuth;
    document.getElementById('ant-az-val').textContent = cfg.azimuth + '°';
    document.getElementById('ant-elevation').value = cfg.elevation;
    document.getElementById('ant-el-val').textContent = cfg.elevation + '°';
    document.getElementById('antenna-modal').classList.add('open');
    drawAntennaPreview();
}
function closeAntennaModal() { document.getElementById('antenna-modal').classList.remove('open'); }
function selectPattern(el, pat) {
    document.querySelectorAll('.antenna-pattern-card').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
    drawAntennaPreview();
}
function selectPol(el, pol) {
    document.querySelectorAll('.pol-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
}
function applyAntennaConfig() {
    const activePattern = document.querySelector('.antenna-pattern-card.active');
    const activePol = document.querySelector('.pol-btn.active');
    state.antennaConfig = {
        pattern: activePattern ? activePattern.dataset.pattern : 'tr38901',
        polarization: activePol ? activePol.dataset.pol : 'cross',
        rows: parseInt(document.getElementById('ant-rows').value) || 8,
        cols: parseInt(document.getElementById('ant-cols').value) || 8,
        azimuth: parseInt(document.getElementById('ant-azimuth').value) || 0,
        elevation: parseInt(document.getElementById('ant-elevation').value) || 0,
    };
    updateAntennaSummary();
    closeAntennaModal();
    // Trigger auto-compute after antenna config change
    if (window._scheduleAutoCompute) window._scheduleAutoCompute();
}
function updateAntennaSummary() {
    const cfg = state.antennaConfig;
    document.getElementById('antenna-summary-text').textContent = `${cfg.rows}x${cfg.cols} ${cfg.pattern}`;
    document.getElementById('stat-array').textContent = `${cfg.rows}x${cfg.cols}`;
}

// Update array element count display
document.getElementById('ant-rows')?.addEventListener('input', () => {
    const r = parseInt(document.getElementById('ant-rows').value) || 1;
    const c = parseInt(document.getElementById('ant-cols').value) || 1;
    document.getElementById('ant-total').textContent = `= ${r * c} elements`;
});
document.getElementById('ant-cols')?.addEventListener('input', () => {
    const r = parseInt(document.getElementById('ant-rows').value) || 1;
    const c = parseInt(document.getElementById('ant-cols').value) || 1;
    document.getElementById('ant-total').textContent = `= ${r * c} elements`;
});

// ─── Export modal ────────────────────────────────────
function openExportModal() { document.getElementById('export-modal').classList.add('open'); }
function closeExportModal() { document.getElementById('export-modal').classList.remove('open'); }

async function exportScene(fmt) {
    closeExportModal();
    const sc = state.currentScene;
    const sceneId = sc && sc.scene_id;
    if (!sceneId) {
        setStatus('error', 'No scene loaded to export');
        return;
    }
    setStatus('loading', `Exporting ${fmt}...`);
    try {
        const resp = await fetch(`/api/scenes/${sceneId}/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ format: fmt }),
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || `Export failed (${resp.status})`);
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${sceneId}.${fmt}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setStatus('ready', `Exported ${sceneId}.${fmt}`);
    } catch (e) {
        setStatus('error', e.message);
    }
}

// ─── Expose to window for onclick handlers ───────────
window._switchView = switchView;
window._switchSidebarTab = switchSidebarTab;
window._createScene = createScene;
window._loadSelectedScene = loadSelectedScene;
window._computeCoverage = computeCoverage;
window._generateAndAnimateRays = generateAndAnimateRays;
function toggleRealistic(el) {
    el.classList.toggle('active');
    state.realistic = el.classList.contains('active');

    // Add/remove warm point light for realistic mode
    const existingPL = scene3d.getObjectByName('realisticLight');
    if (state.realistic && !existingPL) {
        const pl = new THREE.PointLight(0xFFE0B2, 1.5, 20);
        const sc = state.currentScene;
        const w = (sc && sc.room && sc.room.width) || 5;
        const l = (sc && sc.room && sc.room.length) || 4;
        pl.position.set(w / 2, 2.4, l / 2);
        pl.name = 'realisticLight';
        scene3d.add(pl);
    } else if (!state.realistic && existingPL) {
        scene3d.remove(existingPL);
    }

    // Full redraw needed to rebuild walls and furniture materials
    drawSceneViewport();
}

async function deleteSelectedScene() {
    const sceneId = document.getElementById('scene-select').value;
    if (!sceneId) return;
    if (!confirm(`Delete scene "${sceneId}"?`)) return;
    try {
        const resp = await fetch(`/api/scenes/${sceneId}/delete`, { method: 'DELETE' });
        const d = await resp.json();
        if (d.deleted) {
            setStatus('ready', `Deleted ${sceneId}`);
            refreshSceneList();
        } else {
            setStatus('error', d.error || 'Delete failed');
        }
    } catch (e) { setStatus('error', e.message); }
}

window._toggleRealistic = toggleRealistic;
window._deleteSelectedScene = deleteSelectedScene;
window._fetchOSMScene = fetchOSMScene;
window._toggleLayer = toggleLayer;
window._backToOutdoor = backToOutdoor;
window._showSubPanel = showSubPanel;
window._addFurnitureFromSearch = addFurnitureFromSearch;
window._showCategoryVariants = showCategoryVariants;
window._backToCategories = backToCategories;
window._addSpecificModel = addSpecificModel;
window._toggleChat = toggleChat;

// Focus-mode pane toggles — hide the left sidebar or the bottom analytics
// panel to see the viewport / chat unobstructed. State survives reload.
function togglePane(which) {
    const cls = which === 'sidebar' ? 'hide-sidebar' : 'hide-analytics';
    const on = document.body.classList.toggle(cls);
    try { localStorage.setItem(`pane-${which}-hidden`, on ? '1' : '0'); } catch {}
    // The ResizeObserver on .viewport re-fits the Three.js canvas, but the
    // Plotly charts in the bottom panel were laid out at 0×0 while hidden
    // and only re-measure on a window resize — so re-fit them explicitly
    // once the panel is back in the layout.
    if (!on) {
        requestAnimationFrame(() => {
            for (const id of ['ber-chart', 'coverage-hist', 'pdp-chart']) {
                const el = document.getElementById(id);
                if (el && el.data && window.Plotly) {
                    try { Plotly.Plots.resize(el); } catch {}
                }
            }
            window.dispatchEvent(new Event('resize'));
        });
    }
}
window._togglePane = togglePane;
// Restore last state on load
try {
    if (localStorage.getItem('pane-sidebar-hidden') === '1')
        document.body.classList.add('hide-sidebar');
    if (localStorage.getItem('pane-analytics-hidden') === '1')
        document.body.classList.add('hide-analytics');
} catch {}
window._sendChat = sendChat;
window._enterSelectedBuilding = enterSelectedBuilding;
window._openAntennaModal = openAntennaModal;
window._closeAntennaModal = closeAntennaModal;
window._selectPattern = selectPattern;
window._selectPol = selectPol;
window._applyAntennaConfig = applyAntennaConfig;
window._openExportModal = openExportModal;
window._closeExportModal = closeExportModal;
window._exportScene = exportScene;
