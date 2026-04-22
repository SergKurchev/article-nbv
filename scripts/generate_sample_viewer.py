#!/usr/bin/env python3
"""
generate_sample_viewer.py
=========================
Generates interactive HTML viewer for NBV multi-view samples.

Reads all views of:
  - RGB images      (rgb/*.png)
  - Depth maps      (depth/*.npy)   [float32 metres]
  - Segmentation    (masks/*.png)
  - Camera params   (cameras.json)
  - Color map       (color_map.json)

Projects all pixels with valid depth into world-space, merges them,
and writes a standalone visualization.html with:
  - Orbit / Pan / Zoom
  - Mode toggle: RGB | Segmentation | Instances
  - Per-frame camera frustum visualisation
  - Instance legend
  - Stats bar

Usage:
    python scripts/generate_sample_viewer.py dataset/primitives/stage1/sample_00000
    python scripts/generate_sample_viewer.py dataset/primitives/stage2/sample_00001 --stride 2
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rotmat(qx, qy, qz, qw):
    """Unit quaternion → 3×3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def unproject_frame(depth, rgb, mask, fx, fy, cx, cy, cam_pos, cam_target, cam_up,
                    color_to_info, stride=1, max_depth=5.0):
    """
    Project one frame into world-space points using PyBullet view matrix convention.

    Returns ndarray (N, 8): x, y, z, r, g, b, instance_id, category_id
    """
    H, W = depth.shape

    u = np.arange(0, W, stride, dtype=np.float32)
    v = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth[::stride, ::stride].astype(np.float32)
    valid = (Z > 0.001) & (Z < max_depth)

    # Camera coordinates (OpenGL convention: -Z forward, +Y up, +X right)
    X_cam = (uu - cx) * Z / fx
    Y_cam = -(vv - cy) * Z / fy  # Flip Y for OpenGL
    Z_cam = -Z  # Forward is -Z in OpenGL

    # Build camera-to-world transform from PyBullet view parameters
    cam_pos = np.array(cam_pos, dtype=np.float64)
    cam_target = np.array(cam_target, dtype=np.float64)
    cam_up = np.array(cam_up, dtype=np.float64)

    # Camera coordinate frame
    forward = cam_target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, cam_up)
    right = right / (np.linalg.norm(right) + 1e-12)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-12)

    # Camera-to-world rotation matrix
    R = np.column_stack([right, up, -forward])

    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
    pts_world = pts_cam @ R.T + cam_pos

    rgb_s = rgb[::stride, ::stride]
    mask_s = mask[::stride, ::stride]

    # Segmentation labels
    inst_img = np.full(Z.shape, -1, dtype=np.int32)
    cat_img = np.full(Z.shape, -1, dtype=np.int32)

    mr, mg, mb = mask_s[:,:,0], mask_s[:,:,1], mask_s[:,:,2]

    for color, info in color_to_info.items():
        px = (mr == color[0]) & (mg == color[1]) & (mb == color[2])
        inst_img[px] = info["instance_id"]
        cat_img[px] = info["category_id"]

    fv = valid.ravel()
    pts = pts_world.reshape(-1, 3)[fv]
    r = rgb_s[:,:,0].ravel()[fv].astype(np.float32)
    g = rgb_s[:,:,1].ravel()[fv].astype(np.float32)
    b = rgb_s[:,:,2].ravel()[fv].astype(np.float32)
    inst = inst_img.ravel()[fv].astype(np.float32)
    cat = cat_img.ravel()[fv].astype(np.float32)

    return np.column_stack([pts, r, g, b, inst, cat]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Build point cloud from all frames
# ─────────────────────────────────────────────────────────────────────────────

def build_pointcloud(sample_path, cameras, color_map, stride, max_points):
    """Build merged point cloud from all camera views."""

    color_to_info = {}
    for entry in color_map:
        color_tuple = tuple(entry["color"])
        color_to_info[color_tuple] = entry

    chunks = []
    for view_id, cam_data in cameras.items():
        depth_p = sample_path / "depth" / f"{view_id}.npy"
        rgb_p = sample_path / "rgb" / f"{view_id}.png"
        mask_p = sample_path / "masks" / f"{view_id}.png"

        if not depth_p.exists() or not rgb_p.exists():
            print(f"  [skip] view {view_id}: missing files")
            continue

        depth = np.load(str(depth_p))
        rgb = np.array(Image.open(rgb_p).convert("RGB"))
        mask = np.array(Image.open(mask_p).convert("RGB")) if mask_p.exists() else np.zeros_like(rgb)

        intr = cam_data["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

        # Use position directly from cameras.json (this is cam_eye from PyBullet)
        cam_pos = np.array(cam_data["position"], dtype=np.float64)

        # Use target and up from cameras.json if available, otherwise use defaults
        if "target" in cam_data and "up" in cam_data:
            cam_target = np.array(cam_data["target"], dtype=np.float64)
            cam_up = np.array(cam_data["up"], dtype=np.float64)
        else:
            # Fallback for old datasets
            cam_target = np.array([0.5, 0.0, 0.2], dtype=np.float64)
            cam_up = np.array([0, 0, 1], dtype=np.float64)

        chunk = unproject_frame(depth, rgb, mask, fx, fy, cx, cy,
                                cam_pos, cam_target, cam_up,
                                color_to_info, stride=stride)
        chunks.append(chunk)
        print(f"  View {view_id}: {len(chunk):>8,} pts")

    if not chunks:
        return np.zeros((0, 8), dtype=np.float32)

    pts = np.concatenate(chunks, axis=0)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        print(f"  Downsampled to {max_points:,}")

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Camera frustum helper
# ─────────────────────────────────────────────────────────────────────────────

def frustum_lines(cam_data, size=0.06):
    """Return list of [x1,y1,z1, x2,y2,z2] line segments for a frustum."""
    intr = cam_data["intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    W = cx * 2
    H = cy * 2
    half_w = W / 2 / fx * size
    half_h = H / 2 / fy * size

    cam_pos = np.array(cam_data["position"], dtype=np.float64)
    cam_target = np.array([0.5, 0.0, 0.2], dtype=np.float64)
    cam_up = np.array([0, 0, 1], dtype=np.float64)

    # Build camera frame
    forward = cam_target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, cam_up)
    right = right / (np.linalg.norm(right) + 1e-12)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-12)

    R = np.column_stack([right, up, -forward])

    # Four corners at depth=size in camera space (OpenGL convention)
    corners_cam = np.array([
        [-half_w,  half_h, -size],
        [ half_w,  half_h, -size],
        [ half_w, -half_h, -size],
        [-half_w, -half_h, -size],
    ])
    corners_w = corners_cam @ R.T + cam_pos

    origin = cam_pos
    lines = []
    for c in corners_w:
        lines.append([*origin, *c])
    for i in range(4):
        lines.append([*corners_w[i], *corners_w[(i+1) % 4]])
    return np.array(lines, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def compact_float_array(arr, precision=4):
    """Serialize 1-D float array as compact JS TypedArray literal."""
    return "new Float32Array([" + ",".join(f"{v:.{precision}f}" for v in arr) + "])"


def build_html(pts, cameras, color_map, sample_name, stage):
    """Generate interactive HTML viewer."""

    N = len(pts)
    print(f"  Building HTML for {N:,} points ...")

    # Camera frustums
    all_frustum_lines = []
    cam_list = []
    for view_id, cam_data in sorted(cameras.items()):
        all_frustum_lines.append(frustum_lines(cam_data))
        cam_list.append({
            "view_id": view_id,
            "position": cam_data["position"],
            "rotation": cam_data["rotation"]
        })

    frustum_arr = np.concatenate(all_frustum_lines, axis=0) if all_frustum_lines else np.zeros((0, 6))

    # Build instance info for legend
    instance_info = {}
    category_colors = {}

    for entry in color_map:
        inst_id = entry["instance_id"]
        cat_id = entry["category_id"]
        cat_name = entry["category_name"]

        # Generate color for category
        if cat_id not in category_colors:
            if cat_name == "target_object":
                category_colors[cat_id] = [255, 0, 0]  # Red
            elif cat_name == "robot":
                category_colors[cat_id] = [0, 255, 0]  # Green
            elif cat_name == "obstacle":
                category_colors[cat_id] = [0, 0, 255]  # Blue
            else:
                category_colors[cat_id] = [128, 128, 128]  # Gray

        col = category_colors[cat_id]
        instance_info[str(inst_id)] = {
            "category_name": cat_name,
            "category_id": cat_id,
            "hex": "#{:02x}{:02x}{:02x}".format(*col),
        }

    # Count per-category pixels
    cat_counts = {}
    if N > 0:
        unique_cats = np.unique(pts[:, 7])
        for cat in unique_cats:
            if cat >= 0:
                cat_counts[int(cat)] = int(np.sum(pts[:, 7] == cat))

    # Encode point cloud
    if N > 0:
        xs = pts[:, 0].astype(np.float32)
        ys = pts[:, 1].astype(np.float32)
        zs = pts[:, 2].astype(np.float32)
        rs = np.clip(pts[:, 3], 0, 255).astype(np.uint8)
        gs = np.clip(pts[:, 4], 0, 255).astype(np.uint8)
        bs = np.clip(pts[:, 5], 0, 255).astype(np.uint8)
        insts = pts[:, 6].astype(np.int32)
        cats = pts[:, 7].astype(np.int32)
    else:
        xs = ys = zs = np.zeros(0, np.float32)
        rs = gs = bs = np.zeros(0, np.uint8)
        insts = cats = np.zeros(0, np.int32)

    js_xs = compact_float_array(xs)
    js_ys = compact_float_array(ys)
    js_zs = compact_float_array(zs)
    js_rs = "new Uint8Array([" + ",".join(str(v) for v in rs) + "])"
    js_gs = "new Uint8Array([" + ",".join(str(v) for v in gs) + "])"
    js_bs = "new Uint8Array([" + ",".join(str(v) for v in bs) + "])"
    js_insts = "new Int32Array([" + ",".join(str(v) for v in insts) + "])"
    js_cats = "new Int32Array([" + ",".join(str(v) for v in cats) + "])"

    # Frustum line endpoints
    fl = frustum_arr.ravel()
    js_frustum = compact_float_array(fl)

    # Camera positions
    cam_pos_js = json.dumps([[c["position"][0], c["position"][1], c["position"][2]]
                              for c in cam_list])

    seg_palette_js = json.dumps({str(k): list(v) for k, v in category_colors.items()})
    instance_info_js = json.dumps(instance_info, indent=2)

    num_cameras = len(cam_list)
    num_frustum_lines = len(frustum_arr)
    bg_color = "0x0d1117"

    # Stats
    stats_html = f"{N:,} points · {num_cameras} views · Stage {stage}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBV Viewer — {sample_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:#0d1117;color:#e6edf3;
  font-family:'Segoe UI',system-ui,sans-serif;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;
}}
#header{{
  padding:10px 18px;background:#161b22;
  border-bottom:1px solid #30363d;
  display:flex;align-items:center;gap:10px;flex-shrink:0;
}}
#header h1{{font-size:15px;font-weight:600;flex:1;color:#f0f6fc}}
#header .meta{{font-size:11px;color:#8b949e}}
#toolbar{{
  display:flex;align-items:center;gap:6px;flex-wrap:wrap;
  padding:7px 18px;background:#161b22;
  border-bottom:1px solid #30363d;flex-shrink:0;
}}
.btn{{
  padding:5px 13px;border-radius:5px;
  border:1px solid #30363d;background:#21262d;
  color:#e6edf3;font-size:12px;cursor:pointer;
  transition:background .12s,border-color .12s;
  user-select:none;
}}
.btn:hover{{background:#30363d}}
.btn.active{{background:#1f6feb;border-color:#388bfd;color:#fff}}
.sep{{width:1px;height:20px;background:#30363d;margin:0 2px}}
#legend{{
  display:none;flex-wrap:wrap;gap:8px;align-items:center;
  padding:6px 18px;background:#161b22;
  border-bottom:1px solid #30363d;flex-shrink:0;font-size:12px;
}}
.leg-item{{display:flex;align-items:center;gap:5px;color:#8b949e}}
.leg-dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
#main{{flex:1;display:flex;min-height:0}}
#canvas-container{{flex:1;position:relative;overflow:hidden}}
canvas{{width:100%!important;height:100%!important;display:block}}
#info{{
  position:absolute;bottom:10px;left:14px;
  font-size:10px;color:#484f58;pointer-events:none;line-height:1.6;
}}
#loading{{
  position:absolute;inset:0;background:#0d1117;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:12px;font-size:14px;color:#8b949e;z-index:10;
}}
.spinner{{
  width:36px;height:36px;border:3px solid #30363d;
  border-top-color:#1f6feb;border-radius:50%;
  animation:spin .8s linear infinite;
}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
</style>
</head>
<body>

<div id="header">
  <h1>🎯 NBV Point Cloud — {sample_name}</h1>
  <span class="meta">{stats_html}</span>
</div>

<div id="toolbar">
  <button class="btn active" id="btnRGB"  onclick="setMode('rgb')">RGB</button>
  <div class="sep"></div>
  <button class="btn"        id="btnCat"  onclick="setMode('cat')">Category</button>
  <button class="btn"        id="btnInst" onclick="setMode('inst')">Instances</button>
  <div class="sep"></div>
  <button class="btn active" id="btnFrustums" onclick="toggleFrustums()">Cameras</button>
  <button class="btn active" id="btnCamDots"  onclick="toggleCamDots()">Cam Dots</button>
  <div class="sep"></div>
  <button class="btn" onclick="resetCamera()">Reset View</button>
</div>

<div id="legend"></div>

<div id="main">
  <div id="canvas-container">
    <div id="loading">
      <div class="spinner"></div>
      <span>Building point cloud…</span>
    </div>
    <div id="info">
      Scroll: zoom &nbsp;·&nbsp; Left drag: orbit &nbsp;·&nbsp; Right drag / Middle: pan
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
"use strict";

// ── Data ──────────────────────────────────────────────────────────────────────
const XS    = {js_xs};
const YS    = {js_ys};
const ZS    = {js_zs};
const RS    = {js_rs};
const GS    = {js_gs};
const BS    = {js_bs};
const INSTS = {js_insts};
const CATS  = {js_cats};
const N     = XS.length;

const FRUSTUM_SEGS = {js_frustum};
const CAM_POSITIONS = {cam_pos_js};
const SEG_PALETTE   = {seg_palette_js};
const INSTANCE_INFO = {instance_info_js};

// ── Three.js setup ────────────────────────────────────────────────────────────
const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({{antialias:false}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color({bg_color});

const camera = new THREE.PerspectiveCamera(60,
    container.clientWidth / container.clientHeight, 0.001, 50);
camera.position.set(0, 0.5, 1.5);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.screenSpacePanning = true;

// ── Point cloud geometry ──────────────────────────────────────────────────────
const geo = new THREE.BufferGeometry();
const posArr = new Float32Array(N * 3);
const colArr = new Float32Array(N * 3);

for (let i = 0; i < N; i++) {{
    posArr[i*3]   = XS[i]; posArr[i*3+1] = YS[i]; posArr[i*3+2] = ZS[i];
}}

geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
geo.setAttribute('color',    new THREE.BufferAttribute(colArr, 3));

let currentMode = 'rgb';

function applyColors(mode) {{
    for (let i = 0; i < N; i++) {{
        let r, g, b;
        if (mode === 'rgb') {{
            r = RS[i]/255; g = GS[i]/255; b = BS[i]/255;
        }} else if (mode === 'cat') {{
            const val = CATS[i];
            const key = String(val);
            const col = SEG_PALETTE[key] || [80, 80, 80];
            r = col[0]/255; g = col[1]/255; b = col[2]/255;
        }} else if (mode === 'inst') {{
            const inst = INSTS[i];
            if (inst < 0) {{
                r = 0.3; g = 0.3; b = 0.3;
            }} else {{
                const hash = (inst * 2654435761) >>> 0;
                r = ((hash >>> 16) & 0xFF) / 255;
                g = ((hash >>> 8) & 0xFF) / 255;
                b = (hash & 0xFF) / 255;
            }}
        }}
        colArr[i*3]=r; colArr[i*3+1]=g; colArr[i*3+2]=b;
    }}
    geo.attributes.color.needsUpdate = true;
}}

applyColors('rgb');

const ptsMat = new THREE.PointsMaterial({{size:0.003,vertexColors:true,sizeAttenuation:true}});
const ptsObj = new THREE.Points(geo, ptsMat);
scene.add(ptsObj);

// ── Camera frustums ───────────────────────────────────────────────────────────
const frustumGeo = new THREE.BufferGeometry();
const numSegs = FRUSTUM_SEGS.length / 6;
const fposArr = new Float32Array(numSegs * 6);
for (let i = 0; i < FRUSTUM_SEGS.length; i++) fposArr[i] = FRUSTUM_SEGS[i];
frustumGeo.setAttribute('position', new THREE.BufferAttribute(fposArr, 3));
const frustumMat = new THREE.LineBasicMaterial({{color:0xffa040, opacity:0.6, transparent:true}});
const frustumObj = new THREE.LineSegments(frustumGeo, frustumMat);
scene.add(frustumObj);

// ── Camera position dots ──────────────────────────────────────────────────────
const camGeo = new THREE.BufferGeometry();
const camPosFlat = new Float32Array(CAM_POSITIONS.flat());
camGeo.setAttribute('position', new THREE.BufferAttribute(camPosFlat, 3));
const camMat = new THREE.PointsMaterial({{size:0.018, color:0xffa040, sizeAttenuation:true}});
const camDots = new THREE.Points(camGeo, camMat);
scene.add(camDots);

// ── Grid helper ───────────────────────────────────────────────────────────────
const grid = new THREE.GridHelper(1, 10, 0x222222, 0x1a1a1a);
scene.add(grid);

// ── Auto-centre on point cloud ────────────────────────────────────────────────
geo.computeBoundingBox();
const bb  = geo.boundingBox;
const ctr = new THREE.Vector3(); bb.getCenter(ctr);
const sz  = new THREE.Vector3(); bb.getSize(sz);
const maxDim = Math.max(sz.x, sz.y, sz.z);
camera.position.set(ctr.x, ctr.y + maxDim*0.6, ctr.z + maxDim*1.8);
controls.target.copy(ctr);
controls.update();
const defaultCamPos   = camera.position.clone();
const defaultCamTarget = controls.target.clone();

// ── Hide loading ──────────────────────────────────────────────────────────────
setTimeout(() => {{
    document.getElementById('loading').style.display = 'none';
}}, 50);

// ── UI helpers ────────────────────────────────────────────────────────────────
function setMode(mode) {{
    currentMode = mode;
    document.getElementById('btnRGB').classList.toggle('active', mode==='rgb');
    document.getElementById('btnCat').classList.toggle('active', mode==='cat');
    document.getElementById('btnInst').classList.toggle('active', mode==='inst');
    document.getElementById('legend').style.display = (mode==='cat') ? 'flex' : 'none';
    applyColors(mode);
}}

let frustumsVisible = true;
function toggleFrustums() {{
    frustumsVisible = !frustumsVisible;
    frustumObj.visible = frustumsVisible;
    document.getElementById('btnFrustums').classList.toggle('active', frustumsVisible);
}}

let camDotsVisible = true;
function toggleCamDots() {{
    camDotsVisible = !camDotsVisible;
    camDots.visible = camDotsVisible;
    document.getElementById('btnCamDots').classList.toggle('active', camDotsVisible);
}}

function resetCamera() {{
    camera.position.copy(defaultCamPos);
    controls.target.copy(defaultCamTarget);
    controls.update();
}}

// ── Resize ────────────────────────────────────────────────────────────────────
const ro = new ResizeObserver(() => {{
    const w=container.clientWidth, h=container.clientHeight;
    camera.aspect = w/h;
    camera.updateProjectionMatrix();
    renderer.setSize(w,h);
}});
ro.observe(container);

// ── Render loop ───────────────────────────────────────────────────────────────
(function animate(){{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
}})();
</script>
</body>
</html>
"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate NBV HTML viewer for one multi-view sample.")
    ap.add_argument("sample_path", help="Path to sample_XXXXX folder")
    ap.add_argument("--max-points", type=int, default=500000,
                    help="Max total points after downsampling (default 500000)")
    ap.add_argument("--stride", type=int, default=2,
                    help="Stride for pixel sampling (default 2 = every other pixel)")
    args = ap.parse_args()

    sample_path = Path(args.sample_path).resolve()
    if not sample_path.is_dir():
        print(f"[ERROR] Not a directory: {sample_path}")
        sys.exit(1)

    sample_name = sample_path.name

    # Detect stage from path
    stage = "Unknown"
    if "stage1" in str(sample_path):
        stage = "1 (Single Object)"
    elif "stage2" in str(sample_path):
        stage = "2 (Multi-Object)"
    elif "stage3" in str(sample_path):
        stage = "3 (Multi-Object + Obstacles)"

    print(f"\n[Viewer] Building NBV viewer for {sample_name}")
    print(f"  stride={args.stride}  max_points={args.max_points:,}\n")

    cameras_file = sample_path / "cameras.json"
    color_map_file = sample_path / "color_map.json"

    if not cameras_file.exists():
        print(f"[ERROR] Missing cameras.json")
        sys.exit(1)
    if not color_map_file.exists():
        print(f"[ERROR] Missing color_map.json")
        sys.exit(1)

    cameras = json.loads(cameras_file.read_text())
    color_map = json.loads(color_map_file.read_text())

    pts = build_pointcloud(sample_path, cameras, color_map,
                           stride=args.stride, max_points=args.max_points)

    print(f"\n  Total points: {len(pts):,}")
    print("  Generating HTML ...")

    html = build_html(pts, cameras, color_map, sample_name, stage)

    out_path = sample_path / "visualization.html"
    out_path.write_text(html, encoding="utf-8")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone -> {out_path}  ({size_mb:.1f} MB)")
    print("Open in browser: double-click the file or drag into Chrome/Firefox\n")


if __name__ == "__main__":
    main()
