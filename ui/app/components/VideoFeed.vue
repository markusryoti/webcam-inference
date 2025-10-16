<script setup lang="ts">
const video = ref<HTMLVideoElement | null>(null);
const canvas = ref<HTMLCanvasElement | null>(null);

let pc: RTCPeerConnection | null = null;
let localStream: MediaStream | null = null;

function drawPredictions(predictions: Predictions) {
  if (!canvas.value || !video.value) return;

  const ctx = canvas.value.getContext("2d");
  if (!ctx) return;

  // Clear previous drawings
  ctx.clearRect(0, 0, canvas.value.width, canvas.value.height);

  // Set canvas size to match video
  canvas.value.width = video.value.clientWidth;
  canvas.value.height = video.value.clientHeight;

  // Draw each bounding box
  predictions.boxes.forEach((box, i) => {
    const [x1, y1, x2, y2] = box;
    const label = predictions.labels[i];
    const confidence = predictions.confs[i] ?? 0;

    // Calculate box dimensions
    const boxWidth = x2 - x1;
    const boxHeight = y2 - y1;

    // Draw rectangle
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, boxWidth, boxHeight);

    // Draw label background
    ctx.fillStyle = "#00ff00";
    const text = `${label} ${(confidence * 100).toFixed(1)}%`;
    const textWidth = ctx.measureText(text).width;
    ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);

    // Draw text
    ctx.fillStyle = "#000000";
    ctx.font = "14px Arial";
    ctx.fillText(text, x1 + 5, y1 - 5);
  });
}

interface Predictions {
  labels: string[];
  confs: number[];
  // x1, y1, x2, y2
  boxes: Array<[number, number, number, number]>;
}

async function connect() {
  console.log("connect");

  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 360 },
      audio: false,
    });

    if (video.value) {
      video.value.srcObject = localStream;
      await video.value.play().catch(() => {});
    }

    pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    pc.oniceconnectionstatechange = () => {
      console.log("ICE state:", pc?.iceConnectionState);
    };

    const dataChannel = pc.createDataChannel("predictions");
    console.log("Created data channel:", dataChannel.label);

    dataChannel.onopen = () => {
      console.log("Data channel is open");
    };

    dataChannel.onclose = () => {
      console.log("Data channel is closed");
    };

    dataChannel.onmessage = (event) => {
      try {
        const predictions: Predictions = JSON.parse(event.data);
        drawPredictions(predictions);
      } catch (err) {
        console.error("Error parsing prediction:", err);
      }
    };

    const tracks = localStream.getVideoTracks();
    if (tracks[0]) {
      pc.addTrack(tracks[0], localStream);
    } else {
      console.warn("No video tracks found");
    }

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    console.log("Created offer, sending to server...");

    const resp = await fetch("http://localhost:8000/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: pc.localDescription?.sdp,
        type: pc.localDescription?.type,
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error("Server /offer failed: " + text);
    }

    const answer = await resp.json();
    console.log("Received answer from server, setting remote description...");

    await pc.setRemoteDescription(answer);
    console.log("Remote description set successfully");

    console.log("Connected to server, sending video frames.");
  } catch (err) {
    console.error("Error connecting:", err);
  }
}

async function disconnect() {
  console.log("disconnect");

  if (pc) {
    pc.close();
    pc = null;
  }

  if (localStream) {
    localStream.getTracks().forEach((t) => t.stop());
    localStream = null;
  }

  if (video.value) {
    video.value.srcObject = null;
  }

  if (canvas.value) {
    const ctx = canvas.value.getContext("2d");
    ctx?.clearRect(0, 0, canvas.value.width, canvas.value.height);
  }
}
</script>

<template>
  <h1 class="text-2xl mb-8 text-center">Start a video feed</h1>
  <div class="flex flex-col items-center justify-center gap-4">
    <div class="relative">
      <video ref="video" class="border rounded"></video>
      <canvas
        ref="canvas"
        class="absolute top-0 left-0 w-full h-full pointer-events-none"
      ></canvas>
    </div>
    <div class="flex flex-row gap-2 mt-2">
      <UButton @click="connect">Start</UButton>
      <UButton @click="disconnect" variant="subtle">Disconnect</UButton>
    </div>
  </div>
</template>
