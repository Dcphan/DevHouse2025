'use client';

import { useEffect, useRef, useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

type Quality = {
  hasFace: boolean;
  faceCount: number;
  sizeOK: boolean;
  centeredOK: boolean;
  blurOK: boolean;
  lightingOK: boolean;
  score: number; // 0..100
  reasons: string[];
};

export default function App() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null); // draw bbox + landmarks
  const crop224Ref = useRef<HTMLCanvasElement>(null); // 224x224 face crop
  const scratchRef = useRef<HTMLCanvasElement>(null); // offscreen for metrics

  const streamRef = useRef<MediaStream | null>(null);

  // MediaPipe + loop refs
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number>(0);

  // ✅ WebSocket refs/state
  const wsRef = useRef<WebSocket | null>(null);
  const lastSentRef = useRef<number>(0); // throttle
  const [embeddingInfo, setEmbeddingInfo] = useState<{
    dim?: number;
    preview?: number[];
    err?: string;
  } | null>(null);

  const [quality, setQuality] = useState<Quality>({
    hasFace: false,
    faceCount: 0,
    sizeOK: false,
    centeredOK: false,
    blurOK: false,
    lightingOK: false,
    score: 0,
    reasons: ['Camera off'],
  });

  // ---------- WebSocket helpers ----------
  const connectWS = () => {
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const ws = new WebSocket('ws://localhost:8000/ws/embedding');
    wsRef.current = ws;

    ws.onopen = () => console.log('WS connected');
    ws.onclose = () => console.log('WS closed');
    ws.onerror = (e) => console.log('WS error', e);

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);

        if (msg.error) {
          setEmbeddingInfo({ err: msg.error });
          return;
        }

        const emb = Array.isArray(msg.embedding) ? msg.embedding : [];
        setEmbeddingInfo({
          dim: msg.embedding_dim ?? emb.length,
          preview: emb.slice(0, 512),
        });
      } catch (e) {
        setEmbeddingInfo({ err: 'Failed to parse WS message' });
      }
    };
  };

  const disconnectWS = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  const cropCanvasToBase64 = (canvas: HTMLCanvasElement) => {
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    return dataUrl.split(',')[1] ?? '';
  };

  // ---------- MediaPipe init ----------
  const initFaceLandmarker = async () => {
    if (landmarkerRef.current) return;

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    landmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 2,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  };

  // ---------- Helpers ----------
  const stopLoop = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;

    const o = overlayRef.current;
    if (o) o.getContext('2d')?.clearRect(0, 0, o.width, o.height);
  };

  const stopCamera = () => {
    stopLoop();

    // ✅ CLOSE WEBSOCKET HERE
    disconnectWS();

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;

    setIsCameraOn(false);
    setEmbeddingInfo(null);
    setQuality({
      hasFace: false,
      faceCount: 0,
      sizeOK: false,
      centeredOK: false,
      blurOK: false,
      lightingOK: false,
      score: 0,
      reasons: ['Camera off'],
    });
  };

  // Compute bbox (in pixels) from normalized landmarks
  const bboxFromLandmarksPx = (
    landmarks: Array<{ x: number; y: number }>,
    w: number,
    h: number
  ) => {
    let minX = 1,
      minY = 1,
      maxX = 0,
      maxY = 0;
    for (const p of landmarks) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    minX = Math.max(0, Math.min(1, minX));
    minY = Math.max(0, Math.min(1, minY));
    maxX = Math.max(0, Math.min(1, maxX));
    maxY = Math.max(0, Math.min(1, maxY));

    const x = minX * w;
    const y = minY * h;
    const bw = (maxX - minX) * w;
    const bh = (maxY - minY) * h;

    return { x, y, w: bw, h: bh };
  };

  const expandRect = (
    r: { x: number; y: number; w: number; h: number },
    pad: number,
    W: number,
    H: number
  ) => {
    const nx = Math.max(0, r.x - pad);
    const ny = Math.max(0, r.y - pad);
    const nw = Math.min(W - nx, r.w + 2 * pad);
    const nh = Math.min(H - ny, r.h + 2 * pad);
    return { x: nx, y: ny, w: nw, h: nh };
  };

  const laplacianVariance = (img: ImageData) => {
    const { data, width, height } = img;
    const g = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4 + 0];
      const gg = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      g[i] = 0.299 * r + 0.587 * gg + 0.114 * b;
    }

    let sum = 0;
    let sum2 = 0;
    let n = 0;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        const c = g[idx];
        const L =
          -4 * c + g[idx - 1] + g[idx + 1] + g[idx - width] + g[idx + width];
        sum += L;
        sum2 += L * L;
        n++;
      }
    }
    if (n === 0) return 0;
    const mean = sum / n;
    return sum2 / n - mean * mean;
  };

  const meanAndStdLuma = (img: ImageData) => {
    const { data, width, height } = img;
    const N = width * height;
    let sum = 0;
    let sum2 = 0;
    for (let i = 0; i < N; i++) {
      const r = data[i * 4 + 0];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      sum += y;
      sum2 += y * y;
    }
    const mean = sum / N;
    const varr = sum2 / N - mean * mean;
    return { mean, std: Math.sqrt(Math.max(0, varr)) };
  };

  const computeQuality = (
    faceCount: number,
    bbox: { x: number; y: number; w: number; h: number },
    frameW: number,
    frameH: number,
    blurVar: number,
    lumaMean: number,
    lumaStd: number
  ): Quality => {
    const reasons: string[] = [];
    const hasFace = faceCount > 0;

    if (faceCount !== 1)
      reasons.push(faceCount === 0 ? 'No face' : 'Multiple faces');

    const areaRatio = (bbox.w * bbox.h) / (frameW * frameH);
    const sizeOK = areaRatio >= 0.10 && areaRatio <= 0.60;
    if (!sizeOK) reasons.push('Face size not ideal (move closer/farther)');

    const cx = bbox.x + bbox.w / 2;
    const cy = bbox.y + bbox.h / 2;
    const dx = (cx - frameW / 2) / frameW;
    const dy = (cy - frameH / 2) / frameH;
    const centerDist = Math.sqrt(dx * dx + dy * dy);
    const centeredOK = centerDist <= 0.15;
    if (!centeredOK) reasons.push('Center your face');

    const blurOK = blurVar >= 120;
    if (!blurOK) reasons.push('Too blurry (hold still / better focus)');

    const lightingOK = lumaMean >= 70 && lumaMean <= 200 && lumaStd >= 25;
    if (!lightingOK) reasons.push('Lighting not ideal (too dark/bright/flat)');

    let score = 0;
    if (faceCount === 1) score += 25;
    if (sizeOK) score += 20;
    if (centeredOK) score += 20;
    if (blurOK) score += 20;
    if (lightingOK) score += 15;
    score = Math.max(0, Math.min(100, score));

    return {
      hasFace,
      faceCount,
      sizeOK,
      centeredOK,
      blurOK,
      lightingOK,
      score,
      reasons: reasons.length ? reasons : ['OK'],
    };
  };

  const drawOverlay = (
    ctx: CanvasRenderingContext2D,
    frameW: number,
    frameH: number,
    bbox: { x: number; y: number; w: number; h: number } | null,
    landmarks: Array<{ x: number; y: number }> | null
  ) => {
    ctx.clearRect(0, 0, frameW, frameH);

    if (bbox) {
      ctx.lineWidth = 4;
      ctx.strokeStyle = 'lime';
      ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);
    }

    if (landmarks) {
      ctx.fillStyle = 'cyan';
      for (const p of landmarks) {
        const x = p.x * frameW;
        const y = p.y * frameH;
        ctx.beginPath();
        ctx.arc(x, y, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  const cropTo224 = (
    video: HTMLVideoElement,
    rect: { x: number; y: number; w: number; h: number },
    outCanvas: HTMLCanvasElement
  ) => {
    const out = outCanvas.getContext('2d');
    if (!out) return;

    outCanvas.width = 224;
    outCanvas.height = 224;
    out.clearRect(0, 0, 224, 224);
    out.drawImage(video, rect.x, rect.y, rect.w, rect.h, 0, 0, 224, 224);
  };

  // ---------- Main loop ----------
  const startLoop = () => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const crop224 = crop224Ref.current;
    const scratch = scratchRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !overlay || !crop224 || !scratch || !landmarker) return;

    const octx = overlay.getContext('2d');
    if (!octx) return;

    overlay.width = video.videoWidth || 1280;
    overlay.height = video.videoHeight || 720;

    scratch.width = 256;
    scratch.height = 256;

    const tick = () => {
      if (
        !videoRef.current ||
        !overlayRef.current ||
        !crop224Ref.current ||
        !scratchRef.current ||
        !landmarkerRef.current
      )
        return;

      const frameW = overlayRef.current.width;
      const frameH = overlayRef.current.height;

      const now = performance.now();
      const ts = Math.max(now, lastTsRef.current + 0.001);
      lastTsRef.current = ts;

      const res = landmarkerRef.current.detectForVideo(videoRef.current, ts);
      const faces = res.faceLandmarks ?? [];
      const faceCount = faces.length;

      const lm = faceCount > 0 ? faces[0] : null;

      let bbox: { x: number; y: number; w: number; h: number } | null = null;

      if (lm) {
        bbox = bboxFromLandmarksPx(lm as any, frameW, frameH);
        const pad = Math.max(bbox.w, bbox.h) * 0.15;
        bbox = expandRect(bbox, pad, frameW, frameH);

        const sctx = scratchRef.current.getContext('2d');
        if (sctx) {
          sctx.clearRect(0, 0, scratch.width, scratch.height);
          sctx.drawImage(
            videoRef.current,
            bbox.x,
            bbox.y,
            bbox.w,
            bbox.h,
            0,
            0,
            scratch.width,
            scratch.height
          );

          const img = sctx.getImageData(0, 0, scratch.width, scratch.height);
          const blurVar = laplacianVariance(img);
          const { mean: lumaMean, std: lumaStd } = meanAndStdLuma(img);

          const q = computeQuality(
            faceCount,
            bbox,
            frameW,
            frameH,
            blurVar,
            lumaMean,
            lumaStd
          );
          setQuality(q);

          cropTo224(videoRef.current, bbox, crop224Ref.current);

          // ✅ SEND TO BACKEND ONLY WHEN PASS + ~4 FPS
          const passNow = q.score >= 80 && q.faceCount === 1;
          const canSend = now - lastSentRef.current >= 250; // 4 fps

          const ws = wsRef.current;
          if (passNow && canSend && ws && ws.readyState === WebSocket.OPEN) {
            const b64 = cropCanvasToBase64(crop224Ref.current);
            if (b64) {
              ws.send(
                JSON.stringify({
                  track_id: 'face-1',
                  image_b64: b64,
                })
              );
              lastSentRef.current = now;
            }
          }
        }
      } else {
        setQuality({
          hasFace: false,
          faceCount,
          sizeOK: false,
          centeredOK: false,
          blurOK: false,
          lightingOK: false,
          score: 0,
          reasons: faceCount === 0 ? ['No face'] : ['Multiple faces'],
        });
      }

      drawOverlay(
        octx,
        frameW,
        frameH,
        bbox,
        lm ? (lm as any) : null
      );

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
  };

  // ---------- Camera start ----------
  const startCamera = async () => {
    try {
      setError(null);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      const video = videoRef.current;
      if (!video) return;

      video.srcObject = stream;
      streamRef.current = stream;

      video.onloadedmetadata = async () => {
        try {
          await video.play();
          setIsCameraOn(true);

          // ✅ connect WS when camera starts
          connectWS();

          await initFaceLandmarker();
          startLoop();
        } catch {
          setError("Video couldn't start playing. Click the page once, then try again.");
        }
      };
    } catch (e: any) {
      console.error(e);
      setError(e?.message ?? 'Could not start camera.');
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pass = quality.score >= 80 && quality.faceCount === 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-5xl">
        <div className="text-center mb-6">
          <h1 className="text-white mb-2 text-2xl font-semibold">
            Face Capture Pipeline
          </h1>
          <p className="text-slate-400">
            Video → landmarks → quality gate → 224×224 crop → WS → embedding
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* LEFT: video + overlay */}
          <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 overflow-hidden">
            <div className="relative bg-black aspect-video">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-cover ${isCameraOn ? '' : 'hidden'}`}
              />
              <canvas
                ref={overlayRef}
                className={`absolute inset-0 w-full h-full pointer-events-none ${isCameraOn ? '' : 'hidden'}`}
              />
              {!isCameraOn && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                  <Camera className="w-16 h-16 mb-4" />
                  <p>Camera is off</p>
                </div>
              )}
            </div>

            {error && (
              <div className="bg-red-500/10 border-t border-red-500/40 text-red-300 px-4 py-3">
                {error}
              </div>
            )}

            <div className="p-4 flex gap-3 justify-center">
              {!isCameraOn ? (
                <button
                  onClick={startCamera}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  <Camera className="w-5 h-5" />
                  Turn On Camera
                </button>
              ) : (
                <button
                  onClick={stopCamera}
                  className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                >
                  <CameraOff className="w-5 h-5" />
                  Turn Off Camera
                </button>
              )}
            </div>
          </div>

          {/* RIGHT: quality + crop */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-white font-semibold">Quality Gate</h2>
              <span
                className={`text-sm px-2 py-1 rounded ${
                  pass
                    ? 'bg-green-500/20 text-green-300 border border-green-500/40'
                    : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40'
                }`}
              >
                {pass ? 'PASS' : 'ADJUST'}
              </span>
            </div>

            <div className="text-slate-200 text-sm mb-3">
              Score: <span className="font-semibold">{quality.score}</span>/100
            </div>

            <ul className="text-slate-300 text-sm space-y-1 mb-4">
              <li>Faces: {quality.faceCount}</li>
              <li>Size: {quality.sizeOK ? 'OK' : 'No'}</li>
              <li>Centered: {quality.centeredOK ? 'OK' : 'No'}</li>
              <li>Sharpness: {quality.blurOK ? 'OK' : 'No'}</li>
              <li>Lighting: {quality.lightingOK ? 'OK' : 'No'}</li>
            </ul>

            <div className="text-slate-400 text-sm mb-4">
              {quality.reasons.join(' • ')}
            </div>

            <div className="mb-4">
              <h3 className="text-white font-semibold mb-2">224×224 Face Crop</h3>
              <div className="bg-black rounded-lg p-2 inline-block">
                <canvas ref={crop224Ref} className="w-56 h-56 rounded" />
              </div>
            </div>

            {/* ✅ Embedding preview */}
            <div className="mt-2">
              <h3 className="text-white font-semibold mb-2">Backend Embedding</h3>
              {!embeddingInfo && (
                <div className="text-slate-400 text-sm">No embedding yet</div>
              )}
              {embeddingInfo?.err && (
                <div className="text-red-300 text-sm">{embeddingInfo.err}</div>
              )}
              {embeddingInfo?.dim && (
                <div className="text-slate-300 text-sm space-y-1">
                  <div>Dim: {embeddingInfo.dim}</div>
                  <div className="break-all">
                    Preview: [{embeddingInfo.preview?.map((v) => v.toFixed(4)).join(', ')}...]
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* hidden scratch canvas */}
        <canvas ref={scratchRef} className="hidden" />
      </div>
    </div>
  );
}
