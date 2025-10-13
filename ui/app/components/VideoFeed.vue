<script setup lang="ts">
import { ref } from "vue";

let pc: RTCPeerConnection | null = null;
let localStream: MediaStream | null = null;

async function connect() {
  console.log("connect");
  const videoEl = document.querySelector("video") as HTMLVideoElement | null;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    // show locally as before
    if (videoEl) {
      videoEl.srcObject = localStream;
      await videoEl.play().catch(() => {});
    }

    // create peer connection
    pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    // optional: handle ICE candidates locally for debugging
    pc.oniceconnectionstatechange = () => {
      console.log("ICE state:", pc?.iceConnectionState);
    };

    // add the local video track(s) to the connection
    const tracks = localStream.getVideoTracks();
    if (tracks.length > 0) {
      pc.addTrack(tracks[0]!, localStream);
    } else {
      console.warn("No video tracks found");
    }

    // create offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // wait a short while for ICE candidates to gather (optional) or send immediately
    // send offer to server
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
    await pc.setRemoteDescription(answer);
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

  const videoEl = document.querySelector("video") as HTMLVideoElement | null;
  if (videoEl) {
    videoEl.srcObject = null;
  }
}
</script>

<template>
  <h1 class="text-2xl mb-8 text-center">Start a video feed</h1>
  <div class="flex flex-col items-center justify-center gap-4">
    <video class="border rounded"></video>
    <div class="flex flex-row gap-2 mt-2">
      <UButton @click="connect">Start</UButton>
      <UButton @click="disconnect" variant="subtle">Disconnect</UButton>
    </div>
  </div>
</template>
