import * as THREE from "https://esm.sh/three@0.160.0";
import { STLLoader } from "https://esm.sh/three@0.160.0/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const holder = document.getElementById("canvas-holder");
const controlsDiv = document.getElementById("controls");
const metaDiv = document.getElementById("meta");
const drop = document.getElementById("drop");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
camera.position.set(120, 120, 120);
const renderer = new THREE.WebGLRenderer({ antialias: true });
holder.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(1, 1, 1);
scene.add(dir);
const orbit = new OrbitControls(camera, renderer.domElement);
const group = new THREE.Group();
scene.add(group);

function resize() {
  const w = holder.clientWidth, h = holder.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();
(function loop() { requestAnimationFrame(loop); orbit.update(); renderer.render(scene, camera); })();

const loader = new STLLoader();
const meshes = {};

function clearScene() {
  for (const k of Object.keys(meshes)) { group.remove(meshes[k]); delete meshes[k]; }
  controlsDiv.innerHTML = "";
}

function addMesh(role, geometry, colorHex) {
  geometry.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colorHex), transparent: true, opacity: role === "lesao" ? 1.0 : 0.5,
    roughness: 0.7, metalness: 0.0,
  });
  const mesh = new THREE.Mesh(geometry, mat);
  meshes[role] = mesh;
  group.add(mesh);
}

function frameScene() {
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  group.position.sub(center);
  camera.position.set(size, size, size);
  camera.near = size / 100; camera.far = size * 10; camera.updateProjectionMatrix();
  orbit.target.set(0, 0, 0); orbit.update();
}

function buildControls(items) {
  for (const it of items) {
    const row = document.createElement("div"); row.className = "row";
    const label = document.createElement("label");
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = true;
    cb.onchange = () => { if (meshes[it.role]) meshes[it.role].visible = cb.checked; };
    const sw = document.createElement("span"); sw.className = "swatch"; sw.style.background = it.color;
    label.append(cb, sw, document.createTextNode(" " + it.role));
    row.appendChild(label);
    const op = document.createElement("input");
    op.type = "range"; op.min = "0"; op.max = "1"; op.step = "0.05";
    op.value = it.role === "lesao" ? "1" : "0.5";
    op.oninput = () => { if (meshes[it.role]) meshes[it.role].material.opacity = parseFloat(op.value); };
    row.appendChild(op);
    controlsDiv.appendChild(row);
  }
}

// fileMap: role -> ArrayBuffer ; manifest object
function render(manifest, fileMap) {
  clearScene();
  for (const it of manifest.meshes) {
    const buf = fileMap[it.stl];
    if (!buf) { console.warn("STL ausente:", it.stl); continue; }
    addMesh(it.role, loader.parse(buf), it.color);
  }
  frameScene();
  buildControls(manifest.meshes);
  metaDiv.textContent =
    `caso: ${manifest.case_id}\norgão: ${manifest.organ}\n` +
    `coordenadas: ${manifest.coordinate_system}\nestado: ${manifest.regulatory_state}\n` +
    `${manifest.disclaimer || ""}`;
}

// --- Drag & drop of the outputs/ folder (or its files) ---
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hover"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
drop.addEventListener("drop", async (e) => {
  e.preventDefault(); drop.classList.remove("hover");
  const files = [...e.dataTransfer.files];
  const byName = {};
  let manifest = null;
  for (const f of files) {
    const buf = await f.arrayBuffer();
    if (f.name.endsWith(".json")) manifest = JSON.parse(new TextDecoder().decode(buf));
    else byName[f.name] = buf;
  }
  if (!manifest) { alert("Inclua o viewer_manifest.json no que foi arrastado."); return; }
  render(manifest, byName);
});

// --- Optional ?case=<path> when served over http ---
const params = new URLSearchParams(location.search);
const casePath = params.get("case");
if (casePath) {
  (async () => {
    const base = casePath.replace(/\/$/, "");
    const manifest = await (await fetch(`${base}/viewer_manifest.json`)).json();
    const fileMap = {};
    for (const it of manifest.meshes) {
      fileMap[it.stl] = await (await fetch(`${base}/${it.stl}`)).arrayBuffer();
    }
    render(manifest, fileMap);
  })().catch((err) => { console.error(err); alert("Falha ao carregar via ?case: " + err.message); });
}
