from io import BytesIO
from pathlib import Path
from typing import Optional
from uuid import uuid4
import json
import re
import zipfile

import numpy as np
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
TEMPLATE_DIR = ROOT / "templates"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Shirt Print API")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/template-files", StaticFiles(directory=TEMPLATE_DIR), name="template-files")


def safe_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    if not re.fullmatch(r"[a-z0-9_-]{1,80}", name):
        raise HTTPException(status_code=400, detail="Template name can only contain letters, numbers, _ and -")
    return name


def load_image(upload: UploadFile) -> Image.Image:
    return Image.open(BytesIO(upload.file.read())).convert("RGBA")


def load_image_from_url(url: str) -> Image.Image:
    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGBA")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image URL: {e}")


def url_to_filename(url: str) -> str:
    path = url.split("?")[0].rstrip("/").split("/")[-1]
    if not re.fullmatch(r"[A-Za-z0-9_.\- ]{1,120}", path):
        return "design_from_url"
    return path


def resolve_design_image(
    design: UploadFile | None,
    design_url: Optional[str],
) -> tuple[Image.Image, str]:
    if design is not None and design.filename:
        return load_image(design), design.filename
    if design_url:
        return load_image_from_url(design_url), url_to_filename(design_url)
    raise HTTPException(
        status_code=400,
        detail="Either 'design' file upload or 'design_url' field is required",
    )


def remove_background(image: Image.Image) -> Image.Image:
    output = remove(image.convert("RGBA"))
    if isinstance(output, Image.Image):
        return output.convert("RGBA")
    return Image.open(BytesIO(output)).convert("RGBA")


def should_remove_background(image: Image.Image) -> bool:
    if image.mode == "RGBA":
        alpha = np.array(image.getchannel("A"))
        if (alpha < 128).sum() > alpha.size * 0.01:
            return False
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    corners = np.array([arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]])
    corner_spread = float(corners.std(axis=0).mean())
    return corner_spread < 25


def trim_transparent_padding(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    alpha_thresh = alpha.point(lambda p: 0 if p < 10 else p)
    bbox = alpha_thresh.getbbox()
    if not bbox:
        return image
    return image.crop(bbox)


def feather_alpha(image: Image.Image, feather: float) -> Image.Image:
    feather = max(0.0, float(feather))
    if feather <= 0:
        return image.convert("RGBA")
    image = image.convert("RGBA")
    alpha = image.getchannel("A").filter(ImageFilter.GaussianBlur(radius=feather))
    output = image.copy()
    output.putalpha(alpha)
    return output


def feather_canvas_edges(image: Image.Image, feather: float) -> Image.Image:
    feather = max(0.0, float(feather))
    if feather <= 0:
        return image.convert("RGBA")
    image = image.convert("RGBA")
    w, h = image.size
    distance = max(1.0, feather)
    y, x = np.ogrid[:h, :w]
    horizontal_distance = np.minimum(x, w - 1 - x)
    vertical_distance = np.minimum(y, h - 1 - y)
    edge_distance = np.minimum(horizontal_distance, vertical_distance).astype(np.float32)
    fade = np.clip(edge_distance / distance, 0.0, 1.0)
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32)
    alpha = np.clip(alpha * fade, 0, 255).astype(np.uint8)
    output = image.copy()
    output.putalpha(Image.fromarray(alpha, "L"))
    return output


def smooth_design_texture(image: Image.Image, strength: float) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return image.convert("RGBA")
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    rgb = image.convert("RGB")
    radius = 1.0 + 3.0 * strength
    smoothed = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
    # Blend instead of fully replacing so text and line art are not destroyed immediately.
    rgb = Image.blend(rgb, smoothed, 0.55 * strength)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(60 * strength), threshold=4))
    output = rgb.convert("RGBA")
    output.putalpha(alpha)
    return output


def sharpen_design(image: Image.Image, amount: float) -> Image.Image:
    amount = max(0.0, min(2.0, float(amount)))
    if amount <= 0:
        return image.convert("RGBA")
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    rgb = image.convert("RGB")
    sharp = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(120 * amount), threshold=3))
    output = sharp.convert("RGBA")
    output.putalpha(alpha)
    return output


def normalize_design_canvas(image: Image.Image, canvas_size: int) -> Image.Image:
    canvas_size = int(canvas_size)
    if canvas_size <= 0:
        return image.convert("RGBA")
    image = trim_transparent_padding(image.convert("RGBA"))
    scale = min(canvas_size / image.width, canvas_size / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    canvas.alpha_composite(image, ((canvas_size - image.width) // 2, (canvas_size - image.height) // 2))
    return canvas


def prepare_design(
    image: Image.Image,
    remove_bg: bool,
    optimize: bool,
    canvas_size: int,
    edge_feather: float,
    sharpen: float,
    canvas_edge_feather: float = 0,
    smooth_strength: float = 0,
) -> Image.Image:
    if remove_bg and should_remove_background(image):
        design = remove_background(image)
    else:
        design = image.convert("RGBA")
    if edge_feather > 0:
        design = feather_alpha(design, edge_feather)
    if canvas_edge_feather > 0:
        design = feather_canvas_edges(design, canvas_edge_feather)
    design = trim_transparent_padding(design)
    if smooth_strength > 0:
        design = smooth_design_texture(design, smooth_strength)
    if optimize:
        design = normalize_design_canvas(design, canvas_size)
        design = sharpen_design(design, sharpen)
    return design


def apply_opacity(image: Image.Image, opacity: float) -> Image.Image:
    opacity = max(0.0, min(1.0, opacity))
    output = image.copy()
    alpha = output.getchannel("A")
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    output.putalpha(alpha)
    return output


def apply_garment_texture(print_canvas: Image.Image, garment: Image.Image, strength: float) -> Image.Image:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 0:
        return print_canvas

    alpha = print_canvas.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return print_canvas

    print_arr = np.asarray(print_canvas.convert("RGBA"), dtype=np.float32)
    garment_luma = np.asarray(garment.convert("L"), dtype=np.float32)
    alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0

    x1, y1, x2, y2 = bbox
    region_luma = garment_luma[y1:y2, x1:x2]
    region_alpha = alpha_arr[y1:y2, x1:x2]
    visible = region_alpha > 0.05
    if not np.any(visible):
        return print_canvas

    local_mean = float(region_luma[visible].mean())
    local_mean = max(local_mean, 1.0)

    # Normalize by the print region mean. This keeps dark garments from making the print black,
    # while still preserving relative folds and shadows inside the printed area.
    normalized = garment_luma / local_mean
    factor = 1.0 + (normalized - 1.0) * strength
    factor = np.clip(factor, 0.70, 1.30)

    print_arr[:, :, :3] = np.clip(print_arr[:, :, :3] * factor[:, :, None], 0, 255)
    print_arr[:, :, 3] = np.asarray(alpha, dtype=np.float32)
    return Image.fromarray(print_arr.astype(np.uint8), "RGBA")


def fit_contain(image: Image.Image, width: float, height: float, scale_override: float = 1.0) -> Image.Image:
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    scale = min(width / image.width, height / image.height) * max(0.01, float(scale_override))
    out_w = max(1, int(image.width * scale))
    out_h = max(1, int(image.height * scale))
    return image.resize((out_w, out_h), Image.Resampling.LANCZOS)


def fit_cover(image: Image.Image, width: float, height: float, scale_override: float = 1.0) -> Image.Image:
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    scale = max(width / image.width, height / image.height) * max(0.01, float(scale_override))
    out_w = max(1, int(image.width * scale))
    out_h = max(1, int(image.height * scale))
    resized = image.resize((out_w, out_h), Image.Resampling.LANCZOS)
    left = max(0, (out_w - int(width)) // 2)
    top = max(0, (out_h - int(height)) // 2)
    return resized.crop((left, top, left + int(width), top + int(height)))


def compose_print(
    garment: Image.Image,
    design: Image.Image,
    mask: Image.Image | None,
    x: float,
    y: float,
    width: float,
    height: float,
    rotation: float,
    opacity: float,
    scale: float = 1.0,
    texture_strength: float = 0.0,
) -> Image.Image:
    base = garment.convert("RGBA")
    original = base.copy()

    design_layer = trim_transparent_padding(design.convert("RGBA"))
    design_layer = fit_contain(design_layer, width, height, scale)
    design_layer = apply_opacity(design_layer, opacity)
    if rotation:
        design_layer = design_layer.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    center_x = float(x) + float(width) / 2
    center_y = float(y) + float(height) / 2
    paste_x = int(center_x - design_layer.width / 2)
    paste_y = int(center_y - design_layer.height / 2)

    canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas.alpha_composite(design_layer, (paste_x, paste_y))
    canvas = apply_garment_texture(canvas, original, texture_strength)
    base.alpha_composite(canvas)

    if mask is not None:
        mask_l = mask.convert("L").resize(base.size, Image.Resampling.LANCZOS)
        base = Image.composite(original, base, mask_l)

    return base.convert("RGB")


def template_path(name: str) -> Path:
    return TEMPLATE_DIR / safe_name(name)


def write_export_zip(zip_filename: str, filenames: list[str]) -> None:
    with zipfile.ZipFile(OUTPUT_DIR / zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in filenames:
            archive.write(OUTPUT_DIR / filename, arcname=filename)


def base_url(request: Request) -> str:
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}"


def export_response(
    export_format: str,
    result_filename: str,
    design_filename: str,
    params_filename: str,
    zip_filename: str,
    request: Request,
) -> JSONResponse:
    base = base_url(request)
    if export_format == "zip":
        return JSONResponse(
            {
                "result_url": f"{base}/outputs/{result_filename}",
                "export_zip_url": f"{base}/outputs/{zip_filename}",
            }
        )
    return JSONResponse(
        {
            "result_url": f"{base}/outputs/{result_filename}",
            "processed_design_url": f"{base}/outputs/{design_filename}",
            "params_url": f"{base}/outputs/{params_filename}",
        }
    )


def load_template(name: str) -> dict:
    folder = template_path(name)
    config_path = folder / "template.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return json.loads(config_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Shirt Print - 模板管理</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f1f5f9;color:#1e293b}

/* ===== Sidebar ===== */
.sidebar{position:fixed;left:0;top:0;bottom:0;width:230px;background:#1e293b;color:#cbd5e1;display:flex;flex-direction:column;z-index:100}
.sidebar .logo{padding:22px 20px;font-size:17px;font-weight:700;color:#fff;border-bottom:1px solid #334155;letter-spacing:.3px}
.sidebar nav{padding:10px 0;flex:1}
.sidebar nav a{display:flex;align-items:center;gap:10px;padding:11px 20px;color:#94a3b8;text-decoration:none;cursor:pointer;font-size:14px;transition:.12s}
.sidebar nav a:hover{background:#334155;color:#fff}
.sidebar nav a.active{background:#2563eb;color:#fff}
.sidebar .nav-icon{font-size:17px;width:22px;text-align:center}
.sidebar .sidebar-footer{padding:12px 20px;border-top:1px solid #334155;font-size:12px;color:#64748b}

/* ===== Content ===== */
.content{margin-left:230px;min-height:100vh}
.topbar{background:#fff;padding:14px 32px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:50}
.topbar h1{font-size:19px;font-weight:600}
.topbar .status{display:flex;align-items:center;gap:6px;font-size:13px;color:#64748b}
.dot{width:8px;height:8px;border-radius:50%;background:#10b981}
.view{display:none;padding:28px 32px}
.view.active{display:block}

/* ===== Buttons ===== */
.btn{padding:8px 16px;border:0;border-radius:8px;cursor:pointer;font-size:14px;font-weight:500;transition:.12s;text-decoration:none;display:inline-block}
.btn-primary{background:#2563eb;color:#fff}
.btn-primary:hover{background:#1d4ed8}
.btn-secondary{background:#f1f5f9;color:#475569;border:1px solid #e2e8f0}
.btn-secondary:hover{background:#e2e8f0}
.btn-danger{background:#fef2f2;color:#dc2626}
.btn-danger:hover{background:#fee2e2}
.btn-sm{padding:5px 10px;font-size:13px}

/* ===== Template cards ===== */
.tmpl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:20px}
.tmpl-card{background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;transition:.15s}
.tmpl-card:hover{box-shadow:0 8px 24px #00000012}
.tmpl-card .thumb{width:100%;height:220px;object-fit:contain;background:#f8fafc;border-bottom:1px solid #e2e8f0}
.tmpl-card .body{padding:14px 16px}
.tmpl-card .name{font-weight:600;font-size:15px;margin-bottom:6px}
.tmpl-card .params{font-size:12px;color:#64748b;display:flex;flex-wrap:wrap;gap:6px}
.badge{padding:2px 7px;border-radius:5px;font-size:11px;font-weight:500}
.badge-mask{background:#ede9fe;color:#6d28d9}
.badge-nomask{background:#f1f5f9;color:#94a3b8}
.badge-blue{background:#dbeafe;color:#1d4ed8}
.tmpl-actions{display:flex;gap:6px;margin-top:12px}

/* ===== Form controls ===== */
.panel{background:#fff;border-radius:12px;padding:20px;border:1px solid #e2e8f0;margin-bottom:20px}
.form-group{margin-bottom:14px}
.form-group label{display:block;font-size:13px;font-weight:600;color:#475569;margin-bottom:4px}
.form-group input,.form-group select{width:100%;padding:8px 10px;border:1px solid #cbd5e1;border-radius:8px;font-size:14px;background:#fff}
.form-group input:focus,.form-group select:focus{outline:none;border-color:#2563eb;box-shadow:0 0 0 3px #2563eb1a}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.form-row-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.editor-grid{display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}
canvas{max-width:100%;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0}
.section-title{font-size:13px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.5px;margin:18px 0 10px;padding-bottom:6px;border-bottom:1px solid #e2e8f0}
.section-title:first-child{margin-top:0}

/* ===== Generate ===== */
.gen-result{text-align:center;padding:20px}
.gen-result img{max-width:100%;border-radius:12px;border:1px solid #e2e8f0}
.history-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.history-item{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;cursor:pointer;transition:.12s}
.history-item:hover{border-color:#2563eb;box-shadow:0 4px 12px #0001}
.history-item img{width:100%;height:130px;object-fit:cover;background:#fff}
.history-item .hinfo{padding:8px 10px}
.history-item .hinfo .ht{font-weight:600;font-size:12px;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.history-item .hinfo .hd{color:#94a3b8;font-size:11px;margin-top:2px}

/* ===== API view ===== */
.api-card{background:#fff;border-radius:12px;padding:24px;border:1px solid #e2e8f0;margin-bottom:16px}
.api-card h3{font-size:15px;margin-bottom:12px;color:#1e293b}
.api-method{display:inline-block;padding:2px 8px;border-radius:5px;font-size:12px;font-weight:700;margin-right:8px}
.api-method.post{background:#dbeafe;color:#1d4ed8}
.api-method.get{background:#dcfce7;color:#166534}
.api-method.delete{background:#fee2e2;color:#991b1b}
.api-code{background:#1e293b;color:#e2e8f0;padding:14px 16px;border-radius:8px;font-family:Consolas,monospace;font-size:13px;overflow-x:auto;margin:10px 0;line-height:1.6}
.api-code .k{color:#93c5fd}.api-code .s{color:#86efac}.api-code .c{color:#64748b}
.hint{color:#64748b;font-size:13px;line-height:1.6}
.empty{text-align:center;padding:60px 20px;color:#94a3b8}
.empty .icon{font-size:48px;margin-bottom:12px}
</style>
</head>
<body>

<!-- ===== Sidebar ===== -->
<div class="sidebar">
  <div class="logo">👕 Shirt Print</div>
  <nav>
    <a class="nav-item active" data-view="templates"><span class="nav-icon">📋</span>模板管理</a>
    <a class="nav-item" data-view="editor"><span class="nav-icon">✏️</span>编辑模板</a>
    <a class="nav-item" data-view="generate"><span class="nav-icon">⚡</span>快速生成</a>
    <a class="nav-item" data-view="api"><span class="nav-icon">🔗</span>API 接口</a>
  </nav>
  <div class="sidebar-footer">本地服务 · 端口 7861</div>
</div>

<!-- ===== Content ===== -->
<div class="content">
  <div class="topbar">
    <h1 id="pageTitle">模板管理</h1>
    <div class="status"><span class="dot"></span> 服务正常</div>
  </div>

  <!-- ===== View: Templates ===== -->
  <div id="view-templates" class="view active">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
      <p class="hint">管理印花模板，编辑后可供 n8n 批量调用生成效果图</p>
      <button class="btn btn-primary" onclick="newTemplate()">+ 新建模板</button>
    </div>
    <div id="tmplGrid" class="tmpl-grid"></div>
  </div>

  <!-- ===== View: Editor ===== -->
  <div id="view-editor" class="view">
    <div class="editor-grid">
      <!-- Left: Form -->
      <div>
        <div class="panel">
          <div class="section-title">基本信息</div>
          <div class="form-group">
            <label>加载已有模板</label>
            <select id="editorTemplateSelect" onchange="if(this.value)loadTemplateIntoEditor(this.value)"></select>
          </div>
          <div class="form-group">
            <label>模板名称</label>
            <input id="templateName" value="new_template" />
          </div>
          <div class="form-group">
            <label>衣服底图</label>
            <input id="garmentFile" type="file" accept="image/*" />
          </div>
          <div class="form-group">
            <label>遮挡 Mask（可选）</label>
            <input id="maskFile" type="file" accept="image/*" />
          </div>
          <div class="form-group">
            <label>测试图案（可选预览）</label>
            <input id="previewDesignFile" type="file" accept="image/*" />
          </div>
        </div>

        <div class="panel">
          <div class="section-title">编辑模式</div>
          <div class="form-row">
            <div class="form-group">
              <label>模式</label>
              <select id="editMode">
                <option value="box">编辑贴图框</option>
                <option value="mask">画遮挡 Mask</option>
              </select>
            </div>
            <div class="form-group">
              <label>Mask 画笔大小</label>
              <input id="brushSize" type="number" value="24" min="2" max="200" />
            </div>
          </div>
          <div style="display:flex;gap:6px">
            <button class="btn btn-secondary btn-sm" id="undoMask" type="button">撤销</button>
            <button class="btn btn-secondary btn-sm" id="clearMask" type="button">清空 Mask</button>
          </div>
        </div>

        <div class="panel">
          <div class="section-title">贴图区域</div>
          <div class="form-row">
            <div class="form-group"><label>X 坐标</label><input id="x" type="number" value="300" /></div>
            <div class="form-group"><label>Y 坐标</label><input id="y" type="number" value="330" /></div>
            <div class="form-group"><label>宽度</label><input id="w" type="number" value="280" /></div>
            <div class="form-group"><label>高度</label><input id="h" type="number" value="340" /></div>
          </div>
          <div class="section-title">效果参数</div>
          <div class="form-row-3">
            <div class="form-group"><label>旋转°</label><input id="rotation" type="number" value="0" step="0.1" /></div>
            <div class="form-group"><label>透明度</label><input id="opacity" type="number" value="0.92" min="0" max="1" step="0.01" /></div>
            <div class="form-group"><label>纹理强度</label><input id="textureStrength" type="number" value="0.35" min="0" max="1" step="0.01" /></div>
            <div class="form-group"><label>边缘羽化</label><input id="edgeFeather" type="number" value="2" min="0" max="10" step="0.1" /></div>
            <div class="form-group"><label>画布羽化</label><input id="canvasEdgeFeather" type="number" value="0" min="0" max="200" step="1" /></div>
            <div class="form-group"><label>图案平滑</label><input id="smoothStrength" type="number" value="0" min="0" max="1" step="0.05" /></div>
          </div>
        </div>

        <div style="display:flex;gap:8px">
          <button class="btn btn-primary" id="saveTemplate" type="button">保存模板</button>
          <button class="btn btn-secondary" id="newTemplate" type="button">新建</button>
        </div>
        <p id="saveStatus" class="hint" style="margin-top:10px"></p>
      </div>

      <!-- Right: Canvas -->
      <div class="panel">
        <div class="section-title">预览编辑器</div>
        <p class="hint" style="margin-bottom:12px">拖动框内移动位置，拖动右下角缩放。Mask 模式下在遮挡区域涂抹。</p>
        <canvas id="editor" width="768" height="1024"></canvas>
      </div>
    </div>
  </div>

  <!-- ===== View: Generate ===== -->
  <div id="view-generate" class="view">
    <div style="max-width:680px">
      <div class="panel">
        <div class="section-title">选择模板和图案</div>
        <div class="form-group">
          <label>模板</label>
          <select id="templateSelect"></select>
        </div>
        <div class="form-group">
          <label>设计图案</label>
          <input id="designFile" type="file" accept="image/*" />
        </div>
        <div class="form-group">
          <label><input id="removeBg" type="checkbox" style="width:auto;margin-right:6px" checked />自动抠图（去除图案背景）</label>
        </div>
        <button class="btn btn-primary" id="generate" type="button">⚡ 生成成品图</button>
        <p id="generateStatus" class="hint" style="margin-top:10px"></p>
      </div>
      <div id="resultPanel" class="panel gen-result" style="display:none">
        <div class="section-title">生成结果</div>
        <img id="result" alt="生成结果" />
        <div id="exportLinks" style="margin-top:14px"></div>
      </div>
    </div>
    <div class="panel" style="margin-top:20px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <div class="section-title" style="margin:0;border:0;padding:0">📜 历史记录</div>
        <button class="btn btn-danger btn-sm" onclick="clearHistory()">清空</button>
      </div>
      <div id="historyList" class="history-grid"></div>
    </div>
  </div>

  <!-- ===== View: API ===== -->
  <div id="view-api" class="view">
    <div class="api-card">
      <h3>📡 n8n 调用</h3>
      <p class="hint" style="margin-bottom:12px">在 n8n 中添加 HTTP Request 节点，按以下配置调用：</p>
      <div style="margin:8px 0"><span class="api-method post">POST</span><code id="apiBaseUrl">/template-compose</code></div>
      <div class="api-code"><span class="c"># n8n HTTP Request 节点配置</span>
Method: <span class="k">POST</span>
URL: <span class="s">https://你的隧道地址/template-compose</span>
Body: <span class="k">Form-Data Multipart</span>

<span class="c"># 字段（二选一）</span>
template_name: <span class="s">black_hoodie_front</span>
design: <span class="s">&lt;图片文件 Binary&gt;</span>
design_url: <span class="s">https://example.com/design.png</span>
remove_bg: <span class="s">true</span></div>
      <p class="hint">返回 JSON：<code>{"result_url":"/outputs/xxx.jpg", "processed_design_url":"...", "params_url":"..."}</code></p>
    </div>

    <div class="api-card">
      <h3>📋 全部接口</h3>
      <div style="margin:10px 0"><span class="api-method get">GET</span><code>/templates</code> <span class="hint">— 列出所有模板</span></div>
      <div style="margin:10px 0"><span class="api-method post">POST</span><code>/templates</code> <span class="hint">— 保存模板</span></div>
      <div style="margin:10px 0"><span class="api-method post">POST</span><code>/template-compose</code> <span class="hint">— 用模板+图案生成效果图（n8n 主接口）</span></div>
      <div style="margin:10px 0"><span class="api-method post">POST</span><code>/compose</code> <span class="hint">— 自由坐标合成（无需模板）</span></div>
      <div style="margin:10px 0"><span class="api-method post">POST</span><code>/remove-bg</code> <span class="hint">— 单独抠图</span></div>
      <div style="margin:10px 0"><span class="api-method get">GET</span><code>/health</code> <span class="hint">— 健康检查</span></div>
    </div>
  </div>
</div>

<script>
/* ===== fetch with retry (handles 530/502/503 tunnel flaps) ===== */
async function fetchRetry(url, opts, tries = 6, delay = 3000) {
  let res;
  for (let i = 0; i < tries; i++) {
    try {
      res = await fetch(url, opts);
      if (![530, 502, 503, 504].includes(res.status)) return res;
    } catch (e) {
      res = null;
    }
    await new Promise(r => setTimeout(r, delay));
  }
  return res || fetch(url, opts);
}

/* ===== View switching ===== */
const titles = {templates:'模板管理',editor:'编辑模板',generate:'快速生成',api:'API 接口'};
function switchView(name) {
  document.querySelectorAll('.nav-item').forEach(a => a.classList.toggle('active', a.dataset.view === name));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name));
  document.getElementById('pageTitle').textContent = titles[name] || '';
  if (name === 'templates') renderTemplateGrid();
}
document.querySelectorAll('.nav-item').forEach(a => a.addEventListener('click', () => switchView(a.dataset.view)));

/* ===== Template cards ===== */
async function renderTemplateGrid() {
  const res = await fetchRetry('/templates');
  const data = await res.json();
  const grid = document.getElementById('tmplGrid');
  if (!data.templates.length) {
    grid.innerHTML = '<div class="empty"><div class="icon">📋</div>暂无模板，点击「新建模板」创建</div>';
    return;
  }
  grid.innerHTML = data.templates.map(t => `
    <div class="tmpl-card">
      <img class="thumb" src="${t.garment}?t=${Date.now()}" alt="${t.name}" />
      <div class="body">
        <div class="name">${t.name}</div>
        <div class="params">
          <span class="badge ${t.has_mask?'badge-mask':'badge-nomask'}">${t.has_mask?'有遮挡':'无遮挡'}</span>
          <span class="badge badge-blue">${Math.round(t.print_area.width)}×${Math.round(t.print_area.height)}</span>
          <span class="badge badge-blue">透明度 ${(t.opacity??0.9).toFixed(2)}</span>
          <span class="badge badge-blue">纹理 ${(t.texture_strength??0).toFixed(2)}</span>
        </div>
        <div class="tmpl-actions">
          <button class="btn btn-primary btn-sm" onclick="editTemplate('${t.name}')">编辑</button>
          <button class="btn btn-secondary btn-sm" onclick="genWithTemplate('${t.name}')">生成</button>
          <button class="btn btn-danger btn-sm" onclick="deleteTemplate('${t.name}')">删除</button>
        </div>
      </div>
    </div>`).join('');
}
function editTemplate(name) { switchView('editor'); loadTemplateIntoEditor(name); }
function genWithTemplate(name) { switchView('generate'); document.getElementById('templateSelect').value = name; }
async function deleteTemplate(name) {
  if (!confirm('确定删除模板「' + name + '」？此操作不可撤销。')) return;
  const res = await fetchRetry('/templates/' + encodeURIComponent(name), {method:'DELETE'});
  if (res.ok) renderTemplateGrid(); else alert('删除失败');
}

/* ===== Canvas Editor (existing logic) ===== */
const canvas = document.getElementById('editor');
const ctx = canvas.getContext('2d');
const inputs = ['x','y','w','h','rotation','opacity','textureStrength','edgeFeather','canvasEdgeFeather','smoothStrength'].reduce((acc,id)=>{acc[id]=document.getElementById(id);return acc},{});
let garmentImg=null, previewDesign=null, garmentFile=null, maskFile=null;
let maskCanvas=document.createElement('canvas'), maskCtx=maskCanvas.getContext('2d');
let hasMaskDrawing=false, maskHistory=[], loadedTemplateName=null;
let dragMode=null, last=null, displayScale=1;

function frame(){return{x:+inputs.x.value,y:+inputs.y.value,w:+inputs.w.value,h:+inputs.h.value,rotation:+inputs.rotation.value,opacity:+inputs.opacity.value,textureStrength:+inputs.textureStrength.value,edgeFeather:+inputs.edgeFeather.value,canvasEdgeFeather:+inputs.canvasEdgeFeather.value,smoothStrength:+inputs.smoothStrength.value}}
function setFrame(f){inputs.x.value=Math.round(f.x);inputs.y.value=Math.round(f.y);inputs.w.value=Math.round(f.w);inputs.h.value=Math.round(f.h)}
function loadImg(file,cb){const img=new Image();img.onload=()=>cb(img);img.src=URL.createObjectURL(file)}
function loadImgUrl(url,cb){const img=new Image();img.onload=()=>cb(img);img.src=url+'?t='+Date.now()}
function drawContainDesign(img,f){const scale=Math.min(f.w/img.width,f.h/img.height);const dw=img.width*scale,dh=img.height*scale;ctx.save();ctx.globalAlpha=f.opacity;ctx.translate((f.x+f.w/2)*displayScale,(f.y+f.h/2)*displayScale);ctx.rotate(f.rotation*Math.PI/180);ctx.drawImage(img,-dw*displayScale/2,-dh*displayScale/2,dw*displayScale,dh*displayScale);ctx.restore()}
function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!garmentImg){ctx.fillStyle='#94a3b8';ctx.font='16px sans-serif';ctx.fillText('请先上传衣服底图',24,40);return}
  const maxW=Math.min(820,garmentImg.width);displayScale=maxW/garmentImg.width;
  canvas.width=garmentImg.width*displayScale;canvas.height=garmentImg.height*displayScale;
  ctx.drawImage(garmentImg,0,0,canvas.width,canvas.height);
  const f=frame();
  if(previewDesign)drawContainDesign(previewDesign,f);
  if(hasMaskDrawing){ctx.save();ctx.globalAlpha=0.35;ctx.drawImage(maskCanvas,0,0,canvas.width,canvas.height);ctx.restore()}
  ctx.save();ctx.translate((f.x+f.w/2)*displayScale,(f.y+f.h/2)*displayScale);ctx.rotate(f.rotation*Math.PI/180);
  ctx.strokeStyle='#2563eb';ctx.lineWidth=3;ctx.strokeRect(-f.w*displayScale/2,-f.h*displayScale/2,f.w*displayScale,f.h*displayScale);
  ctx.fillStyle='#2563eb';ctx.fillRect(f.w*displayScale/2-8,f.h*displayScale/2-8,16,16);ctx.restore();
}
function resetMaskCanvas(){if(!garmentImg)return;maskCanvas.width=garmentImg.width;maskCanvas.height=garmentImg.height;maskCtx.clearRect(0,0,maskCanvas.width,maskCanvas.height);hasMaskDrawing=false;maskHistory=[]}
function pushMaskHistory(){if(!garmentImg)return;maskHistory.push(maskCtx.getImageData(0,0,maskCanvas.width,maskCanvas.height));if(maskHistory.length>30)maskHistory.shift()}
function undoMask(){const p=maskHistory.pop();if(!p)return;maskCtx.putImageData(p,0,0);const px=maskCtx.getImageData(0,0,maskCanvas.width,maskCanvas.height).data;hasMaskDrawing=false;for(let i=3;i<px.length;i+=4){if(px[i]>0){hasMaskDrawing=true;break}}draw()}
function drawMaskLine(from,to){const size=Math.max(2,+document.getElementById('brushSize').value||24);maskCtx.save();maskCtx.strokeStyle='white';maskCtx.lineWidth=size;maskCtx.lineCap='round';maskCtx.lineJoin='round';maskCtx.beginPath();maskCtx.moveTo(from.x,from.y);maskCtx.lineTo(to.x,to.y);maskCtx.stroke();maskCtx.restore();hasMaskDrawing=true}
function maskBlob(){return new Promise(r=>maskCanvas.toBlob(r,'image/png'))}
function pointer(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*(garmentImg.width/r.width),y:(e.clientY-r.top)*(garmentImg.height/r.height)}}

canvas.addEventListener('mousedown',e=>{if(!garmentImg)return;const p=pointer(e);const f=frame();if(document.getElementById('editMode').value==='mask'){pushMaskHistory();dragMode='mask';last=p;drawMaskLine(p,p);draw();return}const near=Math.abs(p.x-(f.x+f.w))<24&&Math.abs(p.y-(f.y+f.h))<24;const inside=p.x>=f.x&&p.x<=f.x+f.w&&p.y>=f.y&&p.y<=f.y+f.h;dragMode=near?'resize':inside?'move':null;last=p});
window.addEventListener('mousemove',e=>{if(!dragMode||!last)return;const p=pointer(e);const f=frame();const dx=p.x-last.x,dy=p.y-last.y;if(dragMode==='mask'){drawMaskLine(last,p);last=p;draw();return}if(dragMode==='move'){f.x+=dx;f.y+=dy}if(dragMode==='resize'){f.w=Math.max(20,f.w+dx);f.h=Math.max(20,f.h+dy)}setFrame(f);last=p;draw()});
window.addEventListener('mouseup',()=>{dragMode=null;last=null});
Object.values(inputs).forEach(input=>input.addEventListener('input',draw));

document.getElementById('garmentFile').addEventListener('change',e=>{garmentFile=e.target.files[0];if(garmentFile)loadImg(garmentFile,img=>{garmentImg=img;resetMaskCanvas();draw()})});
document.getElementById('maskFile').addEventListener('change',e=>{maskFile=e.target.files[0]||null;if(maskFile&&garmentImg){loadImg(maskFile,img=>{maskCanvas.width=garmentImg.width;maskCanvas.height=garmentImg.height;maskCtx.clearRect(0,0,maskCanvas.width,maskCanvas.height);maskCtx.drawImage(img,0,0,maskCanvas.width,maskCanvas.height);hasMaskDrawing=true;maskHistory=[];draw()})}});
document.getElementById('previewDesignFile').addEventListener('change',e=>{const f=e.target.files[0];if(f)loadImg(f,img=>{previewDesign=img;draw()})});
document.getElementById('clearMask').addEventListener('click',()=>{resetMaskCanvas();maskFile=null;document.getElementById('maskFile').value='';draw()});
document.getElementById('undoMask').addEventListener('click',undoMask);
window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'&&document.getElementById('editMode').value==='mask'){e.preventDefault();undoMask()}});

/* ===== Template save/load ===== */
async function loadTemplates(){
  const res=await fetchRetry('/templates');const data=await res.json();
  const selects=[document.getElementById('templateSelect'),document.getElementById('editorTemplateSelect')];
  selects.forEach(select=>{const cur=select.value;select.innerHTML='';data.templates.forEach(t=>{const opt=document.createElement('option');opt.value=t.name;opt.textContent=t.name;select.appendChild(opt)});if(cur)select.value=cur});
}
function newTemplate(){
  loadedTemplateName=null;garmentImg=null;garmentFile=null;maskFile=null;previewDesign=null;hasMaskDrawing=false;
  document.getElementById('garmentFile').value='';document.getElementById('maskFile').value='';document.getElementById('previewDesignFile').value='';
  document.getElementById('templateName').value='new_template';
  inputs.x.value=300;inputs.y.value=330;inputs.w.value=280;inputs.h.value=340;inputs.rotation.value=0;inputs.opacity.value=0.92;inputs.textureStrength.value=0.35;inputs.edgeFeather.value=2;inputs.canvasEdgeFeather.value=0;inputs.smoothStrength.value=0;
  ctx.clearRect(0,0,canvas.width,canvas.height);draw();
  document.getElementById('saveStatus').textContent='新建模板：请上传衣服底图并设置贴图框';
  switchView('editor');
}
document.getElementById('newTemplate').addEventListener('click',newTemplate);

async function loadTemplateIntoEditor(name){
  if(!name)return;
  const res=await fetchRetry('/templates/'+encodeURIComponent(name));
  if(!res.ok){document.getElementById('saveStatus').textContent=await res.text();return}
  const t=await res.json();loadedTemplateName=t.name;garmentFile=null;maskFile=null;
  document.getElementById('templateName').value=t.name;
  inputs.x.value=t.print_area.x;inputs.y.value=t.print_area.y;inputs.w.value=t.print_area.width;inputs.h.value=t.print_area.height;
  inputs.rotation.value=t.rotation||0;inputs.opacity.value=t.opacity??0.92;inputs.textureStrength.value=t.texture_strength??0.35;
  inputs.edgeFeather.value=t.edge_feather??2;inputs.canvasEdgeFeather.value=t.canvas_edge_feather??0;inputs.smoothStrength.value=t.smooth_strength??0;
  loadImgUrl(t.garment,img=>{garmentImg=img;resetMaskCanvas();if(t.mask){loadImgUrl(t.mask,mi=>{maskCtx.drawImage(mi,0,0,maskCanvas.width,maskCanvas.height);hasMaskDrawing=true;maskHistory=[];draw()})}else{draw()}});
  document.getElementById('saveStatus').textContent='已加载模板：'+t.name;
}
document.getElementById('saveTemplate').addEventListener('click',async()=>{
  if(!garmentFile&&!loadedTemplateName){document.getElementById('saveStatus').textContent='请先上传衣服底图，或加载已有模板';return}
  const fd=new FormData();const newName=document.getElementById('templateName').value;fd.append('name',newName);
  if(garmentFile)fd.append('garment',garmentFile);
  if(loadedTemplateName&&loadedTemplateName!==newName)fd.append('source_template',loadedTemplateName);
  if(hasMaskDrawing){const blob=await maskBlob();fd.append('mask',blob,'mask.png')}else if(maskFile)fd.append('mask',maskFile);
  const f=frame();fd.append('x',f.x);fd.append('y',f.y);fd.append('width',f.w);fd.append('height',f.h);fd.append('rotation',f.rotation);fd.append('opacity',f.opacity);fd.append('texture_strength',f.textureStrength);fd.append('edge_feather',f.edgeFeather);fd.append('canvas_edge_feather',f.canvasEdgeFeather);fd.append('smooth_strength',f.smoothStrength);
  const res=await fetchRetry('/templates',{method:'POST',body:fd});
  if(res.ok){loadedTemplateName=newName;document.getElementById('saveStatus').textContent='✅ 模板已保存'}else{document.getElementById('saveStatus').textContent='❌ '+await res.text()}
  await loadTemplates();
});

/* ===== Generate ===== */
document.getElementById('generate').addEventListener('click',async()=>{
  const design=document.getElementById('designFile').files[0];const name=document.getElementById('templateSelect').value;
  if(!design||!name){document.getElementById('generateStatus').textContent='请选择模板并上传图案';return}
  const fd=new FormData();fd.append('template_name',name);fd.append('design',design);
  fd.append('remove_bg',document.getElementById('removeBg').checked?'true':'false');fd.append('export_format','zip');
  document.getElementById('generateStatus').textContent='生成中，首次抠图需加载模型，请稍等...';
  document.getElementById('resultPanel').style.display='none';
  const res=await fetchRetry('/template-compose',{method:'POST',body:fd});
  if(!res.ok){document.getElementById('generateStatus').textContent='❌ '+await res.text();return}
  const data=await res.json();
  document.getElementById('result').src=data.result_url+'?t='+Date.now();
  document.getElementById('resultPanel').style.display='block';
  document.getElementById('generateStatus').textContent='✅ 生成完成';
  document.getElementById('exportLinks').innerHTML='<a class="btn btn-secondary" href="'+data.export_zip_url+'" download>下载压缩包</a>';
  saveHistory({time:new Date().toLocaleString('zh-CN'),template:name,design:design.name,result_url:data.result_url,zip_url:data.export_zip_url,design_url:data.processed_design_url});
});

/* ===== History (localStorage) ===== */
function getHistory(){try{return JSON.parse(localStorage.getItem('genHistory')||'[]')}catch(e){return[]}}
function saveHistory(item){
  let h=getHistory();
  h.unshift(item);
  if(h.length>60)h=h.slice(0,60);
  localStorage.setItem('genHistory',JSON.stringify(h));
  renderHistory();
}
function renderHistory(){
  const h=getHistory();
  const el=document.getElementById('historyList');
  if(!h.length){el.innerHTML='<p class="hint" style="grid-column:1/-1;text-align:center;padding:20px">暂无历史记录</p>';return}
  el.innerHTML=h.map((it,i)=>`
    <div class="history-item" onclick="viewHistory(${i})">
      <img src="${it.result_url}?t=${Date.now()}" alt="${it.template}" loading="lazy" />
      <div class="hinfo">
        <div class="ht">${it.template}</div>
        <div class="hd">${it.design||''}</div>
        <div class="hd">${it.time}</div>
      </div>
    </div>`).join('');
}
function viewHistory(i){
  const it=getHistory()[i];if(!it)return;
  document.getElementById('result').src=it.result_url+'?t='+Date.now();
  document.getElementById('resultPanel').style.display='block';
  document.getElementById('exportLinks').innerHTML='<a class="btn btn-secondary" href="'+it.zip_url+'" download>下载压缩包</a>';
  document.getElementById('generateStatus').textContent='📷 查看历史：'+it.template+' · '+it.time;
  document.getElementById('resultPanel').scrollIntoView({behavior:'smooth'});
}
function clearHistory(){
  if(!confirm('确定清空全部历史记录？'))return;
  localStorage.removeItem('genHistory');
  renderHistory();
}

/* ===== Init ===== */
loadTemplates();draw();renderTemplateGrid();renderHistory();
</script>
</body>
</html>
        """
    )


@app.get("/templates")
def list_templates():
    templates = []
    for folder in sorted(TEMPLATE_DIR.iterdir()):
        config_path = folder / "template.json"
        if folder.is_dir() and config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            templates.append(config)
    return {"templates": templates}


@app.get("/templates/{name}")
def get_template(name: str):
    return load_template(name)


@app.delete("/templates/{name}")
def delete_template(name: str):
    import shutil
    folder = template_path(name)
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    shutil.rmtree(folder)
    return {"ok": True}


@app.post("/templates")
def save_template(
    name: str = Form(...),
    garment: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
    source_template: Optional[str] = Form(None),
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
    rotation: float = Form(0),
    opacity: float = Form(0.92),
    texture_strength: float = Form(0.35),
    edge_feather: float = Form(2),
    canvas_edge_feather: float = Form(0),
    smooth_strength: float = Form(0),
):
    import shutil
    name = safe_name(name)
    folder = template_path(name)
    folder.mkdir(parents=True, exist_ok=True)

    garment_path = folder / "garment.png"
    if garment and garment.filename:
        load_image(garment).save(garment_path)
    elif garment_path.exists():
        pass
    elif source_template and source_template != name:
        src = template_path(safe_name(source_template))
        src_garment = src / "garment.png"
        if not src_garment.exists():
            raise HTTPException(status_code=400, detail=f"Source template '{source_template}' has no garment to copy")
        shutil.copyfile(src_garment, garment_path)
    else:
        raise HTTPException(status_code=400, detail="Garment image is required for a new template")

    has_mask = bool(mask and mask.filename)
    if has_mask:
        load_image(mask).save(folder / "mask.png")
    elif source_template and source_template != name and not (folder / "mask.png").exists():
        src = template_path(safe_name(source_template))
        if (src / "mask.png").exists():
            shutil.copyfile(src / "mask.png", folder / "mask.png")
            has_mask = True
    elif (folder / "mask.png").exists():
        (folder / "mask.png").unlink()

    config = {
        "name": name,
        "garment": f"/template-files/{name}/garment.png",
        "mask": f"/template-files/{name}/mask.png" if has_mask else None,
        "print_area": {"x": x, "y": y, "width": width, "height": height},
        "rotation": rotation,
        "opacity": opacity,
        "texture_strength": texture_strength,
        "edge_feather": edge_feather,
        "canvas_edge_feather": canvas_edge_feather,
        "smooth_strength": smooth_strength,
        "has_mask": has_mask,
    }
    (folder / "template.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


@app.post("/template-compose")
def template_compose(
    template_name: str = Form(...),
    design: UploadFile | None = File(None),
    design_url: Optional[str] = Form(None),
    remove_bg: bool = Form(True),
    optimize: bool = Form(False),
    canvas_size: int = Form(0),
    edge_feather: Optional[float] = Form(None),
    canvas_edge_feather: Optional[float] = Form(None),
    smooth_strength: Optional[float] = Form(None),
    sharpen: float = Form(0),
    offset_x: float = Form(0),
    offset_y: float = Form(0),
    scale: float = Form(1),
    rotation: Optional[float] = Form(None),
    opacity: Optional[float] = Form(None),
    texture_strength: Optional[float] = Form(None),
    export_format: str = Form("links"),
    request: Request = None,
):
    config = load_template(template_name)
    folder = template_path(template_name)
    garment_img = Image.open(folder / "garment.png").convert("RGBA")
    mask_img = Image.open(folder / "mask.png").convert("RGBA") if (folder / "mask.png").exists() else None
    design_source_img, source_filename = resolve_design_image(design, design_url)
    design_img = prepare_design(
        design_source_img,
        remove_bg,
        optimize,
        canvas_size,
        config.get("edge_feather", 2) if edge_feather is None else edge_feather,
        sharpen,
        config.get("canvas_edge_feather", 0) if canvas_edge_feather is None else canvas_edge_feather,
        config.get("smooth_strength", 0) if smooth_strength is None else smooth_strength,
    )

    area = config["print_area"]
    result = compose_print(
        garment_img,
        design_img,
        mask_img,
        area["x"] + offset_x,
        area["y"] + offset_y,
        area["width"],
        area["height"],
        config["rotation"] if rotation is None else rotation,
        config["opacity"] if opacity is None else opacity,
        scale,
        config.get("texture_strength", 0.0) if texture_strength is None else texture_strength,
    )
    export_id = uuid4().hex
    base_name = f"{safe_name(config['name'])}-{export_id}"
    result_filename = f"result-{base_name}.jpg"
    design_filename = f"processed-design-{base_name}.png"
    params_filename = f"design-params-{base_name}.json"
    zip_filename = f"export-{base_name}.zip"

    result.save(OUTPUT_DIR / result_filename, quality=95)
    design_img.save(OUTPUT_DIR / design_filename)

    params = {
        "template_name": config["name"],
        "source_design_filename": source_filename,
        "design_url": design_url,
        "remove_bg": remove_bg,
        "optimize": optimize,
        "canvas_size": canvas_size,
        "edge_feather": config.get("edge_feather", 2) if edge_feather is None else edge_feather,
        "canvas_edge_feather": config.get("canvas_edge_feather", 0) if canvas_edge_feather is None else canvas_edge_feather,
        "smooth_strength": config.get("smooth_strength", 0) if smooth_strength is None else smooth_strength,
        "sharpen": sharpen,
        "print_area": {
            "x": area["x"] + offset_x,
            "y": area["y"] + offset_y,
            "width": area["width"],
            "height": area["height"],
        },
        "offset_x": offset_x,
        "offset_y": offset_y,
        "scale": scale,
        "rotation": config["rotation"] if rotation is None else rotation,
        "opacity": config["opacity"] if opacity is None else opacity,
        "texture_strength": config.get("texture_strength", 0.0) if texture_strength is None else texture_strength,
        "outputs": {
            "result": result_filename,
            "processed_design": design_filename,
            "params": params_filename,
        },
    }
    (OUTPUT_DIR / params_filename).write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    write_export_zip(zip_filename, [result_filename, design_filename, params_filename])

    return export_response(export_format, result_filename, design_filename, params_filename, zip_filename, request)


@app.post("/compose")
def compose(
    garment: UploadFile = File(...),
    design: UploadFile | None = File(None),
    design_url: Optional[str] = Form(None),
    mask: UploadFile | None = File(None),
    x: int = Form(300),
    y: int = Form(330),
    width: int = Form(260),
    height: int = Form(320),
    rotation: float = Form(0),
    opacity: float = Form(0.92),
    texture_strength: float = Form(0.35),
    remove_bg: bool = Form(False),
    optimize: bool = Form(False),
    canvas_size: int = Form(0),
    edge_feather: float = Form(2),
    canvas_edge_feather: float = Form(0),
    smooth_strength: float = Form(0),
    sharpen: float = Form(0),
    export_format: str = Form("links"),
    request: Request = None,
):
    garment_img = load_image(garment)
    design_source_img, source_filename = resolve_design_image(design, design_url)
    design_img = prepare_design(design_source_img, remove_bg, optimize, canvas_size, edge_feather, sharpen, canvas_edge_feather, smooth_strength)
    mask_img = load_image(mask) if mask and mask.filename else None
    result = compose_print(garment_img, design_img, mask_img, x, y, width, height, rotation, opacity, 1.0, texture_strength)

    export_id = uuid4().hex
    result_filename = f"result-{export_id}.jpg"
    design_filename = f"processed-design-{export_id}.png"
    params_filename = f"design-params-{export_id}.json"
    zip_filename = f"export-{export_id}.zip"

    result.save(OUTPUT_DIR / result_filename, quality=95)
    design_img.save(OUTPUT_DIR / design_filename)
    params = {
        "source_design_filename": source_filename,
        "design_url": design_url,
        "remove_bg": remove_bg,
        "optimize": optimize,
        "canvas_size": canvas_size,
        "edge_feather": edge_feather,
        "canvas_edge_feather": canvas_edge_feather,
        "smooth_strength": smooth_strength,
        "sharpen": sharpen,
        "print_area": {"x": x, "y": y, "width": width, "height": height},
        "rotation": rotation,
        "opacity": opacity,
        "texture_strength": texture_strength,
        "outputs": {
            "result": result_filename,
            "processed_design": design_filename,
            "params": params_filename,
        },
    }
    (OUTPUT_DIR / params_filename).write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    write_export_zip(zip_filename, [result_filename, design_filename, params_filename])

    return export_response(export_format, result_filename, design_filename, params_filename, zip_filename, request)


@app.post("/remove-bg")
def remove_bg_endpoint(image: UploadFile = File(...), request: Request = None):
    image_img = load_image(image)
    result = prepare_design(image_img, True, False, 0, 2, 0)
    filename = f"cutout-{uuid4().hex}.png"
    path = OUTPUT_DIR / filename
    result.save(path)
    base = base_url(request)
    return JSONResponse({"url": f"{base}/outputs/{filename}", "filename": filename})


@app.get("/outputs/{filename}")
def output(filename: str):
    return FileResponse(OUTPUT_DIR / filename)
