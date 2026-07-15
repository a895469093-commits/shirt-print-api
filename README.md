---
title: Shirt Print API
emoji: 👕
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Shirt Print API

Local free tool for creating garment print templates and compositing design images onto garments.
It can remove the background from a design image locally with `rembg`.

## Run

```powershell
cd D:\AI\shirt-print-api
.\run-local.ps1
```

Open:

```text
http://127.0.0.1:7861
```

The local script listens on `0.0.0.0:7861`, so other devices can access it through this Windows machine's LAN IP:

```text
http://<windows-lan-ip>:7861
```

If Windows Firewall blocks access, open PowerShell as Administrator and run:

```powershell
cd D:\AI\shirt-print-api
.\open-firewall.ps1
```

For public internet access, forward TCP port `7861` on the router to this Windows machine, or expose `http://127.0.0.1:7861` with a tunnel such as Cloudflare Tunnel/ngrok. Use the public forwarded/tunnel URL for both the web page and n8n.

n8n HTTP Request URL for local/LAN deployment:

```text
POST http://<windows-lan-ip>:7861/template-compose
```

n8n HTTP Request URL for public deployment:

```text
POST https://<your-public-domain-or-tunnel>/template-compose
```

## Docker / Olares Deploy

Build and run locally or in an Olares Docker environment:

```powershell
cd D:\AI\shirt-print-api
docker compose up -d --build
```

Open:

```text
http://<olares-host-ip>:7861
```

Persistent data:

- `./templates` is mounted to `/app/templates` for saved templates.
- `./outputs` is mounted to `/app/outputs` for generated images.
- `./model-cache` is mounted to `/data/u2net` for the `rembg` model cache.

If n8n runs on the same Docker network as this service, call:

```text
http://shirt-print-api:7861/template-compose
```

If n8n is outside this compose network, call:

```text
http://<olares-host-ip>:7861/template-compose
```

n8n `HTTP Request` node settings:

- Method: `POST`
- URL: `http://shirt-print-api:7861/template-compose` or `http://<olares-host-ip>:7861/template-compose`
- Body Content Type: `Form-Data Multipart`
- Send Binary Data: enabled for the `design` file field
- Form fields: `template_name`, `remove_bg`, and optional tuning fields such as `scale`, `offset_x`, `offset_y`, `opacity`, `texture_strength`
- File field name: `design`

## Olares Kubernetes Deploy Without Docker Command

If the Olares terminal has `kubectl` but no `docker`, use:

```bash
kubectl version --client
```

Upload this project folder to Olares, then edit:

```text
k8s/shirt-print-api.yaml
```

Replace this path with the actual Olares project path:

```text
/REPLACE_WITH_OLARES_PATH/shirt-print-api
```

Apply the deployment:

```bash
kubectl apply -f k8s/shirt-print-api.yaml
```

Check status:

```bash
kubectl get pods -l app=shirt-print-api
kubectl logs -f deploy/shirt-print-api
```

Open the website from outside the cluster:

```text
http://<olares-host-ip>:30861
```

Call from n8n if n8n runs inside the same Kubernetes cluster and namespace:

```text
http://shirt-print-api:7861/template-compose
```

If n8n runs in another namespace, use:

```text
http://shirt-print-api.default.svc.cluster.local:7861/template-compose
```

## Template Workflow

Use the web page to create a template:

1. Upload a garment image.
2. Optionally upload an occlusion mask for hoodie strings, zippers, hands, or folds.
3. Drag and resize the print box on the garment.
4. Set rotation and opacity.
5. Set texture strength to let garment folds and shadows show through the print.
6. Set edge feather to soften the cutout boundary.
7. Optionally set canvas edge feather to fade hard straight borders from incomplete source images.
8. Optionally set design smoothing to reduce oil-paint, grain, and material texture.
9. Save the template.

After that, n8n only needs the template name and a design image.

## Mask Rule

The optional mask controls occlusion.

- White mask area: restored from the original garment, shown above the print.
- Black mask area: normal printed result.

Use this for hoodie strings, zippers, folds, hands, or any object that should cover the print.

## n8n Endpoint

Recommended batch endpoint:

```text
POST http://127.0.0.1:7861/template-compose
```

Fields:

- `template_name`: saved template name
- `design`: image file (optional if `design_url` is provided)
- `design_url`: image URL (optional if `design` file is provided). The server downloads it with a 30s timeout, follows redirects, and converts to RGBA. Use this when n8n only has a public image link.
- `remove_bg`: optional boolean, default `true`
- `edge_feather`: optional override. If omitted, the template setting is used.
- `canvas_edge_feather`: optional override. Fades original image borders before trimming.
- `smooth_strength`: optional override, `0` to `1`. Reduces oil-paint/grain texture.
- `optimize`: optional boolean, default `false`
- `canvas_size`: optional output design canvas size, default `0` disabled
- `sharpen`: optional sharpen amount, default `0` disabled
- `offset_x`: optional fine tuning
- `offset_y`: optional fine tuning
- `scale`: optional fine tuning, default `1`
- `rotation`: optional override
- `opacity`: optional override
- `texture_strength`: optional override, `0` to `1`
- `export_format`: optional, default `links`. Use `zip` when the website needs one downloadable archive.

Default API response for n8n:

```json
{
  "result_url": "/outputs/result-black_hoodie_front-xxxx.jpg",
  "processed_design_url": "/outputs/processed-design-black_hoodie_front-xxxx.png",
  "params_url": "/outputs/design-params-black_hoodie_front-xxxx.json"
}
```

Website ZIP response when `export_format=zip`:

```json
{
  "result_url": "/outputs/result-black_hoodie_front-xxxx.jpg",
  "export_zip_url": "/outputs/export-black_hoodie_front-xxxx.zip"
}
```

ZIP contents:

- Final effect image JPG.
- Processed transparent design PNG after background removal, feathering, trimming, smoothing, and other enabled preprocessing.
- JSON design parameters for the generated image.

The service removes transparent padding, then fits the design into the saved print area with contain logic:

```text
scale = min(area_width / design_width, area_height / design_height)
```

Legacy free-position endpoint:

POST multipart form-data to:

```text
http://127.0.0.1:7861/compose
```

Fields:

- `garment`: image file
- `design`: image file, preferably transparent PNG (optional if `design_url` is provided)
- `design_url`: image URL (optional if `design` file is provided)
- `mask`: optional image file
- `remove_bg`: optional boolean, set `true` to remove the design background before composing
- `optimize`: optional boolean, default `false`
- `canvas_size`: optional, default `0` disabled
- `edge_feather`: optional, default `2`
- `canvas_edge_feather`: optional, default `0`
- `smooth_strength`: optional, default `0`
- `sharpen`: optional, default `0`
- `export_format`: optional, default `links`. Use `zip` to return one downloadable archive.
- `x`: print left coordinate
- `y`: print top coordinate
- `width`: print width in pixels
- `height`: print height in pixels
- `rotation`: degrees
- `opacity`: `0` to `1`
- `texture_strength`: `0` to `1`, uses garment luminance to darken the print over folds and shadows

Response:

```json
{
  "result_url": "/outputs/result-xxxx.jpg",
  "processed_design_url": "/outputs/processed-design-xxxx.png",
  "params_url": "/outputs/design-params-xxxx.json"
}
```

## Background Removal Endpoint

POST multipart form-data to:

```text
http://127.0.0.1:7861/remove-bg
```

Fields:

- `image`: image file

Response:

```json
{"url":"/outputs/cutout-xxxx.png","filename":"cutout-xxxx.png"}
```

The first background-removal request downloads the `rembg` model to `D:\AI\.cache\u2net`.

The background removal endpoint returns a transparent PNG with edge feathering. Upscaling and sharpening are disabled by default.
