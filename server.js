const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const bodyParser = require("body-parser");
const faceapi = require("@vladmandic/face-api");
const canvas = require("canvas");
const cors = require("cors");

// Confirm TensorFlow backend (should be 'tensorflow' when using tfjs-node)
try {
  console.log("TFJS backend:", tf.getBackend());
} catch (_) {
  // no-op
}

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: "10mb" }));

// simple health check endpoint for container orchestration
app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

// Patch face-api to use node-canvas and Node 18+ global fetch
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ fetch: globalThis.fetch, Canvas, Image, ImageData });

const modelPath = process.env.MODEL_PATH || "./models";

async function loadModels() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
}
loadModels();

async function loadImageFromSource(source) {
  if (!source) {
    throw new Error("Image source is required");
  }

  if (source.startsWith("data:image/")) {
    const base64Data = source.split(",")[1] ?? source;
    const buffer = Buffer.from(base64Data, "base64");
    return canvas.loadImage(buffer);
  }

  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  return canvas.loadImage(buffer);
}

async function multipleFaces(imageSource) {
  const image = await loadImageFromSource(imageSource);
  const detections = await faceapi.detectAllFaces(
    image,
    new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })
  );
  return detections.map((detection) => ({
    score: detection.score,
    box: {
      x: detection.box.x,
      y: detection.box.y,
      width: detection.box.width,
      height: detection.box.height,
    },
  }));
}

async function singleFace(imageSource) {
  const image = await loadImageFromSource(imageSource);
  const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
  const detectionWithLandmarks = await faceapi
    .detectSingleFace(image, options)
    .withFaceLandmarks();

  if (!detectionWithLandmarks) {
    return null;
  }

  const { detection, landmarks } = detectionWithLandmarks;
  return {
    score: detection.score,
    box: {
      x: detection.box.x,
      y: detection.box.y,
      width: detection.box.width,
      height: detection.box.height,
    },
    landmarks: landmarks.positions.map((point) => ({ x: point.x, y: point.y })),
  };
}

function averagePoint(indices, points) {
  const total = indices.reduce(
    (acc, idx) => {
      const point = points[idx];
      return {
        x: acc.x + point.x,
        y: acc.y + point.y,
      };
    },
    { x: 0, y: 0 }
  );
  const count = indices.length;
  return { x: total.x / count, y: total.y / count };
}

function distanceBetween(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function analyzeLookingAway(detection) {
  const landmarks = detection.landmarks;

  if (!landmarks || landmarks.length < 68) {
    return {
      lookingAway: false,
      metrics: { reason: "insufficient_landmarks" },
    };
  }

  const leftEyeIndices = [36, 37, 38, 39, 40, 41];
  const rightEyeIndices = [42, 43, 44, 45, 46, 47];
  const noseTipIndex = 30;

  const leftEyeCenter = averagePoint(leftEyeIndices, landmarks);
  const rightEyeCenter = averagePoint(rightEyeIndices, landmarks);
  const noseTip = landmarks[noseTipIndex];
  const midpoint = {
    x: (leftEyeCenter.x + rightEyeCenter.x) / 2,
    y: (leftEyeCenter.y + rightEyeCenter.y) / 2,
  };

  const eyeDistance = distanceBetween(leftEyeCenter, rightEyeCenter);
  const normalizedEyeDistance =
    eyeDistance > 0 ? eyeDistance / detection.box.width : 0;

  const yaw = eyeDistance > 0 ? (noseTip.x - midpoint.x) / eyeDistance : 0;
  const distanceLeft = distanceBetween(noseTip, leftEyeCenter);
  const distanceRight = distanceBetween(noseTip, rightEyeCenter);

  const distanceRatio =
    eyeDistance > 0
      ? Math.abs(distanceLeft - distanceRight) / eyeDistance
      : 0;

  const eyeVerticalSkew =
    eyeDistance > 0
      ? Math.abs(leftEyeCenter.y - rightEyeCenter.y) / eyeDistance
      : 0;

  const thresholds = {
    normalizedEyeDistance: 0.32,
    yaw: 0.15,
    distanceRatio: 0.2,
    eyeVerticalSkew: 0.1,
  };

  const lookingAway =
    normalizedEyeDistance < thresholds.normalizedEyeDistance ||
    Math.abs(yaw) > thresholds.yaw ||
    distanceRatio > thresholds.distanceRatio ||
    eyeVerticalSkew > thresholds.eyeVerticalSkew;

  return {
    lookingAway,
    metrics: {
      normalizedEyeDistance,
      normalizedEyeDistanceThreshold: thresholds.normalizedEyeDistance,
      yaw,
      yawThreshold: thresholds.yaw,
      eyeDistanceRatio: distanceRatio,
      eyeDistanceRatioThreshold: thresholds.distanceRatio,
      eyeVerticalSkew,
      eyeVerticalSkewThreshold: thresholds.eyeVerticalSkew,
    },
  };
}

async function compareFaceImages(referenceImage, capturedImage) {
  const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
  const refImage = await loadImageFromSource(referenceImage);
  const capImage = await loadImageFromSource(capturedImage);

  const refDetection = await faceapi
    .detectSingleFace(refImage, options)
    .withFaceLandmarks()
    .withFaceDescriptor();

  const capDetection = await faceapi
    .detectSingleFace(capImage, options)
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!refDetection || !capDetection) {
    return { error: "Face not detected", status: 400 };
  }

  const faceMatcher = new faceapi.FaceMatcher(refDetection.descriptor, 0.5);
  const bestMatch = faceMatcher.findBestMatch(capDetection.descriptor);

  console.log("Distance:", bestMatch.distance);

  return {
    match: bestMatch.distance <= 0.53,
    similarity: 1 - bestMatch.distance,
  };
}

app.post("/api/compare-face", async (req, res) => {
  try {
    const { capturedImage, referenceImage } = req.body;
    if (!referenceImage || !capturedImage) {
      return res.status(400).json({ error: "referenceImage and capturedImage are required" });
    }

    const result = await compareFaceImages(referenceImage, capturedImage);
    if (result.error) {
      return res.status(result.status).json({ error: result.error });
    }

    return res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Face comparison failed" });
  }
});

app.post("/api/check-face", async (req, res) => {
  try {
    const { capturedImage, referenceImageUrl } = req.body;
    if (!referenceImageUrl || !capturedImage) {
      return res.status(400).json({ error: "referenceImageUrl and capturedImage are required" });
    }

    const result = await compareFaceImages(referenceImageUrl, capturedImage);
    if (result.error) {
      return res.status(result.status).json({ error: result.error });
    }

    return res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Face comparison failed" });
  }
});

app.post("/api/detect-face", async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) {
      return res.status(400).json({ error: "Image is required" });
    }

    const detection = await singleFace(image);
    if (!detection) {
      return res.json({ faceDetected: false });
    }

    const { landmarks: _omit, ...sanitizedDetection } = detection;
    res.json({ faceDetected: true, detection: sanitizedDetection });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Face detection failed" });
  }
});

app.post("/api/multiple-faces", async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) {
      return res.status(400).json({ error: "Image is required" });
    }

    const detections = await multipleFaces(image);
    res.json({ count: detections.length, detections });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Multiple face detection failed" });
  }
});

app.post("/api/looking-away", async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) {
      return res.status(400).json({ error: "Image is required" });
    }

    const detection = await singleFace(image);
    if (!detection) {
      return res.json({
        faceDetected: false,
        lookingAway: false,
        metrics: { reason: "no_face_detected" },
      });
    }

    const analysis = analyzeLookingAway(detection);
    const { landmarks: __omit, ...sanitizedDetection } = detection;
    res.json({
      faceDetected: true,
      lookingAway: analysis.lookingAway,
      metrics: analysis.metrics,
      detection: sanitizedDetection,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Looking away detection failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
