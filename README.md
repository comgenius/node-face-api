# Node Face API

This project exposes an HTTP API for face detection, comparison, and pose analysis using `@vladmandic/face-api` accelerated by `@tensorflow/tfjs-node`. It runs on Express, accepts JSON payloads, and loads neural network model weights from the local `./models` directory.

## Getting Started
- **Prerequisites:** Node.js 18+ (for native fetch support and tfjs-node bindings) and yarn/npm.
- **Install dependencies:** `npm install`
- **Start the API:**
  - `npm start` (runs `server.js`, which exposes health check plus the `/api/check-face` verification endpoint), or
  - `node server.js` (runs the extended API described below).
- **Model files:** The service loads pre-trained weights from `./models`. Ensure that directory ships with the repository or is mounted when running in Docker.

### Configuration
- `PORT` (optional): HTTP port (defaults to `3000`).
- `MODEL_PATH` (optional, when running `server.js`): set to override the default `./models` directory.

### Image Inputs
Each endpoint accepts either:
- A publicly reachable HTTPS URL, or
- A data URI / raw Base64 string (e.g. `"data:image/png;base64,iVBOR..."`).

Large base64 payloads can exceed the default request limit; `body-parser` is already configured with a `10mb` ceiling. Increase the limit if you need to send higher-resolution frames.

## API Overview (server.js)

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/health` | Service status probe. |
| POST | `/api/detect-face` | Detects a single face and returns bounding box + landmarks. |
| POST | `/api/multiple-faces` | Detects all faces in the frame. |
| POST | `/api/looking-away` | Detects a face and estimates whether the subject is looking away. |
| POST | `/api/compare-face` | Compares a captured face to a reference image and returns similarity. |

`server.js` exposes the same `/health` route plus a legacy `/api/check-face` endpoint that mirrors `/api/compare-face` but expects the reference image in the `referenceImageUrl` field.

---

### `GET /health`
- **Success (200):**
  ```json
  { "status": "ok" }
  ```
- Useful for load balancers, Docker health checks, or uptime monitors.

### `POST /api/detect-face`
- **Request body:**
  ```json
  { "image": "https://example.com/person.jpg" }
  ```
- **Success (200):**
  ```json
  {
    "faceDetected": true,
    "detection": {
      "score": 0.9987,
      "box": { "x": 220.5, "y": 135.2, "width": 160.9, "height": 160.1 },
      "landmarks": [
        { "x": 263.3, "y": 200.4 },
        { "x": 245.1, "y": 185.7 }
        // ... 68 landmark points
      ]
    }
  }
  ```
- **No face:** `{"faceDetected": false}`
- **Error (400/500):** Missing image or inference issues.

### `POST /api/multiple-faces`
- **Request body:**
  ```json
  { "image": "data:image/jpeg;base64,..." }
  ```
- **Success (200):**
  ```json
  {
    "count": 3,
    "detections": [
      { "score": 0.99, "box": { "x": 42, "y": 58, "width": 120, "height": 110 } },
      { "score": 0.97, "box": { "x": 222, "y": 61, "width": 118, "height": 109 } }
      // ...
    ]
  }
  ```
- The bounding boxes use pixel coordinates relative to the original image.

### `POST /api/looking-away`
- **Request body:**
  ```json
  { "image": "https://example.com/student.png" }
  ```
- **Success (200):**
  ```json
  {
    "faceDetected": true,
    "lookingAway": false,
    "metrics": {
      "normalizedEyeDistance": 0.41,
      "normalizedEyeDistanceThreshold": 0.32,
      "yaw": 0.06,
      "yawThreshold": 0.15,
      "eyeDistanceRatio": 0.04,
      "eyeDistanceRatioThreshold": 0.2,
      "eyeVerticalSkew": 0.02,
      "eyeVerticalSkewThreshold": 0.1
    },
    "detection": { "...": "same structure as /api/detect-face" }
  }
  ```
- **No face detected:** Responds with `lookingAway: false` and a `metrics.reason` describing the failure (`"no_face_detected"` or `"insufficient_landmarks"`).
- Thresholds are tuned for frontal faces with moderate head movement; adjust `analyzeLookingAway` in `server.js` if you need stricter criteria.

### `POST /api/compare-face`
- **Request body:**
  ```json
  {
    "referenceImage": "https://example.com/reference.jpg",
    "capturedImage": "https://example.com/captured.jpg"
  }
  ```
  Both fields can be HTTPS URLs or base64 payloads. This endpoint fetches and decodes each image, extracts descriptors, and performs a similarity comparison.
- **Success (200):**
  ```json
  {
    "match": true,
    "similarity": 0.82
  }
  ```
  `match` is `true` when the computed distance is below 0.53 (configurable in code). `similarity` is `1 - distance`, so higher values reflect closer matches.
- **No face detected:** Returns `400` with `{ "error": "Face not detected" }`.

### `POST /api/check-face` (server.js)
- **Request body:**
  ```json
  {
    "referenceImageUrl": "https://example.com/reference.jpg",
    "capturedImage": "data:image/png;base64,iVBOR..."
  }
  ```
- **Success (200):**
  ```json
  {
    "match": false,
    "similarity": 0.43
  }
  ```
- This endpoint is maintained for backward compatibility with older clients that pass the reference image in the `referenceImageUrl` field.

## Error Handling
- Unexpected inference failures return `500` with `{ "error": "..." }`. Check server logs for stack traces.
- Network timeouts or invalid URLs throw `"Failed to fetch image"` errors.
- Responses include minimal metadata to limit payload size; adjust the response builders in `server.js` if additional diagnostics are required.

## Development Notes
- The face models are loaded on startup. On first boot they may take a few seconds to warm up; subsequent requests are faster because weights stay resident in memory.
- TensorFlow operations spin up native threads; ensure you stop the process gracefully (`Ctrl+C`) to release resources.
- When containerizing, copy the `models/` directory into the image or mount it as a volume; see the provided `Dockerfile`/`docker-compose.yml` for scaffolding.

## Testing the API Quickly
Example curl command that compares two images hosted online:
```bash
curl -X POST http://localhost:3000/api/compare-face \
  -H "Content-Type: application/json" \
  -d '{
        "referenceImage": "https://example.com/reference.jpg",
        "capturedImage": "https://example.com/captured.jpg"
      }'
```
Adjust URLs or encode local snapshots as base64 before submitting.

