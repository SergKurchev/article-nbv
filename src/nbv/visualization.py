"""Point-cloud HTML visualizer for NBV episodes.

Reuses backprojection and Three.js HTML generation from ODIN's
generate_isaac_viewer.py (copied here to remove local-path dependency).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config


# ---------------------------------------------------------------------------
# Geometry helpers (copied from generate_isaac_viewer.py)
# ---------------------------------------------------------------------------

def _quat_to_rotmat(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _unproject_frame(
    depth, rgb, seg_mask, fx, fy, cx, cy, R, t,
    stride=4, max_depth=3.0,
    pred_cat_mask=None, pred_inst_mask=None,
):
    """Back-project one RGB-D frame to world-space.
    Output columns (N, 10): x,y,z, r,g,b, gt_cat, gt_inst, pred_cat, pred_inst.
    gt_cat   = GT semantic class id  (seg_mask)
    gt_inst  = GT instance id  (same as gt_cat for now, env doesn't give instance ids)
    pred_cat = ODIN predicted category id  (pred_cat_mask)
    pred_inst= ODIN predicted instance id  (pred_inst_mask)
    """
    H, W = depth.shape
    u = np.arange(0, W, stride, dtype=np.float32)
    v = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    Z = depth[::stride, ::stride].astype(np.float32)
    valid = (Z > 0.001) & (Z < max_depth)

    X_cam = (uu - cx) * Z / fx
    Y_cam = -(vv - cy) * Z / fy  # negate Y to match training convention
    pts_cam = np.stack([X_cam, Y_cam, Z], axis=-1)
    pts_world = pts_cam.reshape(-1, 3) @ R.T + t

    rgb_s = rgb[::stride, ::stride]
    seg_s  = seg_mask[::stride, ::stride]       if seg_mask       is not None else np.full(Z.shape, -1, np.int32)
    pcat_s = pred_cat_mask[::stride, ::stride]  if pred_cat_mask  is not None else np.full(Z.shape, -1, np.int32)
    pins_s = pred_inst_mask[::stride, ::stride] if pred_inst_mask is not None else np.full(Z.shape, -1, np.int32)

    fv = valid.ravel()
    pts = pts_world[fv]
    r    = rgb_s[:, :, 0].ravel()[fv].astype(np.float32)
    g    = rgb_s[:, :, 1].ravel()[fv].astype(np.float32)
    b    = rgb_s[:, :, 2].ravel()[fv].astype(np.float32)
    gt_cat  = seg_s.ravel()[fv].astype(np.float32)
    gt_inst = seg_s.ravel()[fv].astype(np.float32)   # env gives no per-instance GT yet
    pcat    = pcat_s.ravel()[fv].astype(np.float32)
    pins    = pins_s.ravel()[fv].astype(np.float32)
    return np.column_stack([pts, r, g, b, gt_cat, gt_inst, pcat, pins]).astype(np.float32)


def _frustum_lines(cam_dict, size=0.03):
    intr = cam_dict["intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    half_w = cx / fx * size
    half_h = cy / fy * size
    R = _quat_to_rotmat(*cam_dict["rotation"])
    t = np.array(cam_dict["position"], dtype=np.float64)
    corners_cam = np.array([
        [-half_w,  half_h, size], [ half_w,  half_h, size],
        [ half_w, -half_h, size], [-half_w, -half_h, size],
    ])
    corners_w = corners_cam @ R.T + t
    lines = [[*t, *c] for c in corners_w]
    for i in range(4):
        lines.append([*corners_w[i], *corners_w[(i+1) % 4]])
    return np.array(lines, dtype=np.float32)


def _cat_color(cat_id):
    h = (int(cat_id) * 2654435761) % (2**32)
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)


def _compact(arr, prec=4):
    return "new Float32Array([" + ",".join(f"{v:.{prec}f}" for v in arr) + "])"


# ---------------------------------------------------------------------------
# HTML builder (adapted from generate_isaac_viewer.py)
# ---------------------------------------------------------------------------

def _build_html(pts, cam_list, color_map, title):
    N = len(pts)
    seg_palette = {-1: (80, 80, 80)}
    cat_names = {-1: "background"}
    for entry in color_map:
        cat = entry["category_id"]
        seg_palette[cat] = _cat_color(cat)
        cat_names[cat] = entry.get("category_name", f"class_{cat}")

    all_lines = np.concatenate([_frustum_lines(c) for c in cam_list], axis=0) if cam_list else np.zeros((0, 6))

    if N > 0:
        xs = pts[:, 0]; ys = pts[:, 1]; zs = pts[:, 2]
        rs = np.clip(pts[:, 3], 0, 255).astype(np.uint8)
        gs = np.clip(pts[:, 4], 0, 255).astype(np.uint8)
        bs = np.clip(pts[:, 5], 0, 255).astype(np.uint8)
        gt_cats  = pts[:, 6].astype(np.int32)   # GT semantic category
        gt_insts = pts[:, 7].astype(np.int32)   # GT instance id
        pred_cats = pts[:, 8].astype(np.int32)  # ODIN predicted category
        pred_insts= pts[:, 9].astype(np.int32)  # ODIN predicted instance id
    else:
        xs = ys = zs = np.zeros(0, np.float32)
        rs = gs = bs = np.zeros(0, np.uint8)
        gt_cats = gt_insts = pred_cats = pred_insts = np.zeros(0, np.int32)

    js_xs = _compact(xs); js_ys = _compact(ys); js_zs = _compact(zs)
    js_rs = "new Uint8Array([" + ",".join(str(v) for v in rs) + "])"
    js_gs = "new Uint8Array([" + ",".join(str(v) for v in gs) + "])"
    js_bs = "new Uint8Array([" + ",".join(str(v) for v in bs) + "])"
    js_gt_cats   = "new Int32Array([" + ",".join(str(v) for v in gt_cats)   + "])"
    js_gt_insts  = "new Int32Array([" + ",".join(str(v) for v in gt_insts)  + "])"
    js_pred_cats = "new Int32Array([" + ",".join(str(v) for v in pred_cats) + "])"
    js_pred_insts= "new Int32Array([" + ",".join(str(v) for v in pred_insts)+ "])"
    js_frustum = _compact(all_lines.ravel())
    cam_pos_js = json.dumps([[c["position"][0], c["position"][1], c["position"][2]] for c in cam_list])
    seg_palette_js = json.dumps({str(k): list(v) for k, v in seg_palette.items()})

    legend_html = '<div id="legend">\n'
    for cat, name in cat_names.items():
        if cat < 0:
            continue
        col = seg_palette[cat]
        legend_html += (f'  <div class="leg-item"><div class="leg-dot" '
                        f'style="background:rgb({col[0]},{col[1]},{col[2]})"></div>{name}</div>\n')
    legend_html += '</div>'
    stats = f"{N:,} points · {len(cam_list)} cameras"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>NBV Episode — {title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
#header{{padding:10px 18px;background:#161b22;border-bottom:1px solid #30363d;display:flex;align-items:center;gap:10px;flex-shrink:0}}
#header h1{{font-size:15px;font-weight:600;flex:1;color:#f0f6fc}}
#header .meta{{font-size:11px;color:#8b949e}}
#toolbar{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:7px 18px;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0}}
.btn{{padding:5px 13px;border-radius:5px;border:1px solid #30363d;background:#21262d;color:#e6edf3;font-size:12px;cursor:pointer;transition:background .12s;user-select:none}}
.btn:hover{{background:#30363d}} .btn.active{{background:#1f6feb;border-color:#388bfd;color:#fff}}
.sep{{width:1px;height:20px;background:#30363d;margin:0 2px}}
#legend{{display:none;flex-wrap:wrap;gap:8px;align-items:center;padding:6px 18px;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0;font-size:12px}}
.leg-item{{display:flex;align-items:center;gap:5px;color:#8b949e}}
.leg-dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
#main{{flex:1;position:relative;overflow:hidden}}
canvas{{width:100%!important;height:100%!important;display:block}}
#info{{position:absolute;bottom:10px;left:14px;font-size:10px;color:#484f58;pointer-events:none;line-height:1.6}}
#loading{{position:absolute;inset:0;background:#0d1117;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;font-size:14px;color:#8b949e;z-index:10}}
.spinner{{width:36px;height:36px;border:3px solid #30363d;border-top-color:#1f6feb;border-radius:50%;animation:spin .8s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
</style></head><body>
<div id="header"><h1>NBV Point Cloud — {title}</h1><span class="meta">{stats}</span></div>
<div id="toolbar">
  <button class="btn active" id="btnRGB"      onclick="setMode('rgb')">RGB</button>
  <div class="sep"></div>
  <button class="btn"        id="btnGtCat"    onclick="setMode('gt_cat')">GT Category</button>
  <button class="btn"        id="btnGtInst"   onclick="setMode('gt_inst')">Instances (GT)</button>
  <div class="sep"></div>
  <button class="btn"        id="btnPredCat"  onclick="setMode('pred_cat')">pred_cat (ODIN)</button>
  <button class="btn"        id="btnPredInst" onclick="setMode('pred_inst')">pred_inst (ODIN)</button>
  <div class="sep"></div>
  <button class="btn active" id="btnFrustums" onclick="toggleFrustums()">Cameras</button>
  <div class="sep"></div>
  <button class="btn"        onclick="resetCamera()">Reset View</button>
</div>
{legend_html}
<div id="main">
  <div id="loading"><div class="spinner"></div><span>Building point cloud…</span></div>
  <div id="info">Scroll: zoom · Left drag: orbit · Right drag: pan</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
"use strict";
const XS={js_xs},YS={js_ys},ZS={js_zs};
const RS={js_rs},GS={js_gs},BS={js_bs};
const GT_CATS={js_gt_cats},GT_INSTS={js_gt_insts};
const PRED_CATS={js_pred_cats},PRED_INSTS={js_pred_insts};
const N=XS.length;
const FRUSTUM_SEGS={js_frustum};
const CAM_POSITIONS={cam_pos_js};
const SEG_PALETTE={seg_palette_js};

function idColor(id,salt){{
  if(id<0)return [80,80,80];
  const h=((id+salt*1234567)*2654435761)>>>0;
  return [(h>>16)&0xFF,(h>>8)&0xFF,h&0xFF];
}}

const container=document.getElementById('main');
const renderer=new THREE.WebGLRenderer({{antialias:false}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setSize(container.clientWidth,container.clientHeight);
container.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d1117);
const camera=new THREE.PerspectiveCamera(60,container.clientWidth/container.clientHeight,0.001,50);
camera.position.set(0,0.5,1.0);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.dampingFactor=0.07;

const geo=new THREE.BufferGeometry();
const posArr=new Float32Array(N*3),colArr=new Float32Array(N*3);
for(let i=0;i<N;i++){{posArr[i*3]=XS[i];posArr[i*3+1]=YS[i];posArr[i*3+2]=ZS[i];}}
geo.setAttribute('position',new THREE.BufferAttribute(posArr,3));
geo.setAttribute('color',new THREE.BufferAttribute(colArr,3));

const MODES=['rgb','gt_cat','gt_inst','pred_cat','pred_inst'];
const BTN_IDS={{rgb:'btnRGB',gt_cat:'btnGtCat',gt_inst:'btnGtInst',pred_cat:'btnPredCat',pred_inst:'btnPredInst'}};

function applyColors(mode){{
  const showLegend=(mode==='gt_cat'||mode==='pred_cat');
  document.getElementById('legend').style.display=showLegend?'flex':'none';
  for(let i=0;i<N;i++){{
    let r,g,b;
    if(mode==='rgb'){{r=RS[i]/255;g=GS[i]/255;b=BS[i]/255;}}
    else if(mode==='gt_cat'){{const k=String(GT_CATS[i]);const c=SEG_PALETTE[k]||SEG_PALETTE['-1'];r=c[0]/255;g=c[1]/255;b=c[2]/255;}}
    else if(mode==='gt_inst'){{const c=idColor(GT_INSTS[i],0);r=c[0]/255;g=c[1]/255;b=c[2]/255;}}
    else if(mode==='pred_cat'){{const k=String(PRED_CATS[i]);const c=SEG_PALETTE[k]||SEG_PALETTE['-1'];r=c[0]/255;g=c[1]/255;b=c[2]/255;}}
    else{{const c=idColor(PRED_INSTS[i],7);r=c[0]/255;g=c[1]/255;b=c[2]/255;}}
    colArr[i*3]=r;colArr[i*3+1]=g;colArr[i*3+2]=b;
  }}
  geo.attributes.color.needsUpdate=true;
}}
applyColors('rgb');
scene.add(new THREE.Points(geo,new THREE.PointsMaterial({{size:0.003,vertexColors:true,sizeAttenuation:true}})));

const fgeo=new THREE.BufferGeometry();
const fpos=new Float32Array(FRUSTUM_SEGS.length);
for(let i=0;i<FRUSTUM_SEGS.length;i++)fpos[i]=FRUSTUM_SEGS[i];
fgeo.setAttribute('position',new THREE.BufferAttribute(fpos,3));
const frustumObj=new THREE.LineSegments(fgeo,new THREE.LineBasicMaterial({{color:0xffa040,opacity:0.7,transparent:true}}));
scene.add(frustumObj);

const cgeo=new THREE.BufferGeometry();
cgeo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(CAM_POSITIONS.flat()),3));
scene.add(new THREE.Points(cgeo,new THREE.PointsMaterial({{size:0.015,color:0xffa040,sizeAttenuation:true}})));
scene.add(new THREE.GridHelper(1,10,0x222222,0x1a1a1a));

geo.computeBoundingBox();
const bb=geo.boundingBox,ctr=new THREE.Vector3(),sz=new THREE.Vector3();
bb.getCenter(ctr); bb.getSize(sz);
const maxDim=Math.max(sz.x,sz.y,sz.z);
camera.position.set(ctr.x,ctr.y+maxDim*0.8,ctr.z+maxDim*2.0);
controls.target.copy(ctr); controls.update();
const defPos=camera.position.clone(),defTgt=controls.target.clone();
setTimeout(()=>{{document.getElementById('loading').style.display='none';}},50);

let currentMode='rgb';
function setMode(mode){{
  currentMode=mode;
  MODES.forEach(m=>{{const el=document.getElementById(BTN_IDS[m]);if(el)el.classList.toggle('active',m===mode);}});
  applyColors(mode);
}}
let fv=true;
function toggleFrustums(){{fv=!fv;frustumObj.visible=fv;document.getElementById('btnFrustums').classList.toggle('active',fv);}}
function resetCamera(){{camera.position.copy(defPos);controls.target.copy(defTgt);controls.update();}}
new ResizeObserver(()=>{{const w=container.clientWidth,h=container.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);}}).observe(container);
(function animate(){{requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}})();
</script></body></html>"""


# ---------------------------------------------------------------------------
# Episode accumulator
# ---------------------------------------------------------------------------

def _build_color_map():
    return [
        {"instance_id": i, "category_id": i, "category_name": name, "color": list(_cat_color(i))}
        for i, name in enumerate(config.SHAPE_NAMES)
    ]


class EpisodeVisualizer:
    """Accumulates RGB-D frames across an episode and saves an interactive HTML viewer."""

    def __init__(self, output_path: Path, max_points: int = 400_000, stride: int = 4):
        self.output_path = Path(output_path)
        self.max_points = max_points
        self.stride = stride
        self._chunks: list[np.ndarray] = []
        self._cams: list[dict] = []

    def add_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg_mask: np.ndarray | None,
        cam_pos: np.ndarray,
        cam_quat_xyzw: np.ndarray,
        fx: float = None,
        fy: float = None,
        cx: float = None,
        cy: float = None,
        pred_cat_mask: np.ndarray | None = None,
        pred_inst_mask: np.ndarray | None = None,
    ) -> None:
        """
        rgb:            (H, W, 3) uint8
        depth:          (H, W) float32 metres
        seg_mask:       (H, W) int — GT class id per pixel, -1=background (optional)
        cam_quat_xyzw:  (4,) PyBullet (x,y,z,w)
        pred_cat_mask:  (H, W) int — ODIN predicted category id per pixel (optional)
        pred_inst_mask: (H, W) int — ODIN predicted instance id per pixel (optional)
        """
        half = config.IMAGE_SIZE / 2.0
        fx = fx or half; fy = fy or half; cx = cx or half; cy = cy or half

        qx, qy, qz, qw = cam_quat_xyzw
        R = _quat_to_rotmat(qx, qy, qz, qw)
        # OpenGL → OpenCV convention: negate Z column (same fix as generate_isaac_viewer.py)
        R[:, 2] = -R[:, 2]
        t = np.array(cam_pos, dtype=np.float64)

        chunk = _unproject_frame(
            depth, rgb,
            seg_mask if seg_mask is not None else np.full(depth.shape, -1, np.int32),
            fx, fy, cx, cy, R, t,
            stride=self.stride, max_depth=5.0,
            pred_cat_mask=pred_cat_mask,
            pred_inst_mask=pred_inst_mask,
        )
        self._chunks.append(chunk)
        self._cams.append({
            "frame_index": len(self._cams),
            "position": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
            "rotation": [float(qx), float(qy), float(qz), float(qw)],
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        })

    def save(self, episode_id: int) -> None:
        pts = np.concatenate(self._chunks) if self._chunks else np.zeros((0, 8), np.float32)
        if len(pts) > self.max_points:
            idx = np.random.choice(len(pts), self.max_points, replace=False)
            pts = pts[idx]
        color_map = _build_color_map()
        html = _build_html(pts, self._cams, color_map, f"Episode {episode_id}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")
