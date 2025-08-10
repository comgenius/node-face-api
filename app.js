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

let modelPath = "./models";

async function loadModels() {
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
}
loadModels();

app.post("/api/check-face", async (req, res) => {
  try {
    const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
    const { capturedImage, referenceImageUrl } = req.body;

    const refResponse = await fetch(referenceImageUrl);
    const refArrayBuffer = await refResponse.arrayBuffer();
    const refBuffer = Buffer.from(refArrayBuffer);
    const refImage = await canvas.loadImage(refBuffer);

    const refDetection = await faceapi
      .detectSingleFace(refImage, options)
      .withFaceLandmarks()
      .withFaceDescriptor();

    // const base64Data = capturedImage.replace(/^data:image\/png;base64,/, "");
    // const capturedBuffer = Buffer.from(base64Data, "base64");
    // const capImage = await canvas.loadImage(capturedBuffer);
    const capResponse = await fetch(capturedImage);
    const capArrayBuffer = await capResponse.arrayBuffer();
    const capBuffer = Buffer.from(capArrayBuffer);
    const capImage = await canvas.loadImage(capBuffer);

    const capDetection = await faceapi
      .detectSingleFace(capImage, options)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!refDetection || !capDetection) {
      return res.status(400).json({ error: "Face not detected" });
    }

    const faceMatcher = new faceapi.FaceMatcher(refDetection.descriptor, 0.5);
    const bestMatch = faceMatcher.findBestMatch(capDetection.descriptor);

    console.log("Distance:", bestMatch.distance);

    return res.json({
      match: bestMatch.distance <= 0.53,
      similarity: 1 - bestMatch.distance,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Face comparison failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));