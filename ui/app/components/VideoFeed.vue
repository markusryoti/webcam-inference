<script setup lang="ts">
const video = ref<HTMLVideoElement | null>(null);
const incomingVideo = ref<HTMLVideoElement | null>(null);

let pc: RTCPeerConnection | null = null;
let localStream: MediaStream | null = null;

async function connect() {
  console.log("connect");

  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: true,
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

    // Set up incoming track handler BEFORE creating offer
    pc.ontrack = (event) => {
      console.log("Received remote track:", {
        kind: event.track.kind,
        id: event.track.id,
        readyState: event.track.readyState,
        streamsCount: event.streams.length,
      });

      if (event.track.kind === "video" && incomingVideo.value) {
        const stream = event.streams[0] || new MediaStream([event.track]);
        console.log("Setting incoming video source:", {
          streamId: stream.id,
          tracksCount: stream.getTracks().length,
        });

        incomingVideo.value.srcObject = stream;
        incomingVideo.value.play().catch((err) => {
          console.error("Error playing incoming video:", err);
        });

        console.log("Incoming video element ready");
      } else {
        console.log("Skipping track:", event.track.kind);
      }
    };

    const tracks = localStream.getVideoTracks();
    if (tracks.length > 0) {
      pc.addTrack(tracks[0]!, localStream);
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

  if (incomingVideo.value) {
    incomingVideo.value.srcObject = null;
  }
}
</script>

<template>
  <h1 class="text-2xl mb-8 text-center">Start a video feed</h1>
  <div class="flex flex-col items-center justify-center gap-4">
    <div class="flex gap-4">
      <video ref="video" class="border rounded"></video>
      <video ref="incomingVideo" class="border rounded" autoplay muted></video>
    </div>
    <div class="flex flex-row gap-2 mt-2">
      <UButton @click="connect">Start</UButton>
      <UButton @click="disconnect" variant="subtle">Disconnect</UButton>
    </div>
  </div>
</template>
