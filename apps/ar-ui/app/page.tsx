'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { ConversationSuggestions } from './components/ConversationSuggestions';
import { MetaGlassesOverlay } from './components/MetaGlassesOverlay';
import { PersonRecognitionCard } from './components/PersonRecognitionCard';
import { SummarizePanel } from './components/SummarizePanel';

type OrchestratorPayload = {
  active_person_id?: string | null;
  stm?: { utterances?: Array<{ speaker?: string; utterance?: string }> };
  recent_topics?: { name?: string; ideas?: string[] };
  person_context?: {
    person_id?: string | null;
    person_name?: string | null;
    last_conversation_summary?: string | null;
    shared_interest?: string | null;
    facts?: Array<{ key?: string; value?: string }>;
  };
  coach?: { hint?: string; followups?: string[] };
  memory?: { stored?: boolean; items?: any[] };
  signals?: Array<{ type?: string; [key: string]: any }>;
};

type IdentityInfo = {
  identity?: string | null;
  identityName?: string | null;
  bestMatch?: string | null;
  bestMatchName?: string | null;
  score?: number;
  smoothScore?: number;
  enrolled?: boolean;
  newPersonId?: string | null;
  newPersonName?: string | null;
};

const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE ?? 'ws://localhost:8000';

const formatDuration = (startMs: number) => {
  const elapsed = Math.max(0, Date.now() - startMs);
  const mins = Math.floor(elapsed / 60000);
  const secs = Math.floor((elapsed % 60000) / 1000);
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${secs.toString().padStart(2, '0')}s`;
};

export default function ARApp() {
  const [showPersonCard, setShowPersonCard] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<'starting' | 'on' | 'error'>('starting');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [coachHint, setCoachHint] = useState('');
  const [coachFollowups, setCoachFollowups] = useState<string[]>([]);
  const [recentTopics, setRecentTopics] = useState<{ name?: string; ideas?: string[] }>({});
  const [personContext, setPersonContext] = useState<OrchestratorPayload['person_context']>({});
  const [stm, setStm] = useState<Array<{ speaker?: string; utterance?: string }>>([]);
  const [identityInfo, setIdentityInfo] = useState<IdentityInfo | null>(null);
  const [activePersonId, setActivePersonId] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [audioStatus, setAudioStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('connecting');
  const [audioError, setAudioError] = useState<string | null>(null);

  const embeddingWSRef = useRef<WebSocket | null>(null);
  const audioWSRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string>(`session_${Math.random().toString(36).slice(2, 10)}`);
  const summaryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const suggestionTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sessionStartRef = useRef<number>(Date.now());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioSamplesRef = useRef<number[]>([]);

  const derivedIsNewPerson = useMemo(() => {
    if (identityInfo?.identity === 'UNKNOWN') return true;
    if (identityInfo?.newPersonId) return true;
    if (!identityInfo?.identityName && !personContext?.person_name) return true;
    return false;
  }, [identityInfo, personContext]);

  const personData = useMemo(() => {
    const interests: string[] = [];
    if (personContext?.shared_interest) interests.push(personContext.shared_interest);
    if (Array.isArray(recentTopics?.ideas)) interests.push(...recentTopics.ideas.slice(0, 4));
    if (!interests.length) interests.push('Listening for interests');

    const lastTopic =
      recentTopics?.name ||
      personContext?.last_conversation_summary ||
      null;

    return {
      name: identityInfo?.identityName || personContext?.person_name || 'Unknown person',
      firstMet: derivedIsNewPerson ? 'Just now' : 'This session',
      relationship: personContext?.shared_interest
        ? `Shared: ${personContext.shared_interest}`
        : 'Active conversation',
      lastTopic,
      interests,
    };
  }, [derivedIsNewPerson, identityInfo, personContext, recentTopics]);

  const summaryData = useMemo(() => {
    const utterances = Array.isArray(stm) ? stm.slice(-5) : [];
    const topicsDiscussed = utterances.map((u, i) => ({
      time: `T${i + 1}`,
      topic: u.speaker ? `${u.speaker}` : 'Speaker',
      detail: u.utterance || '',
    }));

    const nextStepsSource = coachFollowups.length
      ? coachFollowups
      : Array.isArray(recentTopics?.ideas)
        ? recentTopics.ideas
        : [];

    return {
      duration: formatDuration(sessionStartRef.current),
      topicsDiscussed: topicsDiscussed.length
        ? topicsDiscussed
        : [
            {
              time: '-',
              topic: 'Conversation warming up',
              detail: 'Keep chatting to populate the summary.',
            },
          ],
      sentiment: coachHint ? 'Engaged' : 'Pending',
      nextSteps: nextStepsSource.length
        ? nextStepsSource
        : ['Keep the conversation going', 'Ask an open question'],
    };
  }, [coachFollowups, coachHint, recentTopics, stm]);

  useEffect(() => {
    let cancelled = false;
    const startCamera = async () => {
      try {
        setCameraStatus('starting');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setCameraStatus('on');
        setCameraError(null);
      } catch (e: any) {
        setCameraStatus('error');
        setCameraError(e?.message ?? 'Unable to start camera');
      }
    };
    if (typeof navigator !== 'undefined' && navigator.mediaDevices?.getUserMedia) {
      startCamera();
    } else {
      setCameraStatus('error');
      setCameraError('Camera API not available in this environment.');
    }
    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
    };
  }, []);

  const handleOrchestratorPayload = (payload: OrchestratorPayload | null) => {
    if (!payload || typeof payload !== 'object') return;
    if (payload.active_person_id && payload.active_person_id !== activePersonId) {
      setActivePersonId(payload.active_person_id);
      setShowPersonCard(true);
    }

    setCoachHint(payload.coach?.hint || '');
    const cleanFollowups = Array.isArray(payload.coach?.followups)
      ? payload.coach.followups.filter((f) => typeof f === 'string')
      : [];
    setCoachFollowups(cleanFollowups);
    if (cleanFollowups.length) {
      setShowSuggestions(true);
      if (suggestionTimeoutRef.current) clearTimeout(suggestionTimeoutRef.current);
      suggestionTimeoutRef.current = setTimeout(() => setShowSuggestions(false), 8000);
    }

    setPersonContext(payload.person_context || {});
    setRecentTopics(payload.recent_topics || {});
    setStm(payload.stm?.utterances || []);

    const hasSummarySignal = (payload.signals || []).some(
      (s) => s?.type === 'meeting_summary_saved' || s?.type === 'recent_topics_updated'
    );
    if (hasSummarySignal) {
      setShowSummary(true);
      if (summaryTimeoutRef.current) clearTimeout(summaryTimeoutRef.current);
      summaryTimeoutRef.current = setTimeout(() => setShowSummary(false), 12000);
    }
  };

  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/embedding`);
    embeddingWSRef.current = ws;

    ws.onopen = () => setConnectionStatus('connected');
    ws.onclose = () => setConnectionStatus('disconnected');
    ws.onerror = () => setConnectionStatus('disconnected');

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'ORCHESTRATOR') {
          handleOrchestratorPayload(msg.payload);
          return;
        }

        if (msg.orchestrator) {
          handleOrchestratorPayload(msg.orchestrator);
        }

        if (msg.identity || msg.identity_name || msg.new_person_id) {
          setIdentityInfo({
            identity: msg.identity ?? null,
            identityName: msg.identity_name ?? null,
            bestMatch: msg.best_match ?? null,
            bestMatchName: msg.best_match_name ?? null,
            score: typeof msg.score === 'number' ? msg.score : undefined,
            smoothScore: typeof msg.smooth_score === 'number' ? msg.smooth_score : undefined,
            enrolled: Boolean(msg.enrolled),
            newPersonId: msg.new_person_id ?? null,
            newPersonName: msg.new_person_name ?? null,
          });
          setShowPersonCard(true);
        }
      } catch {
        // ignore parse errors
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/audio`);
    audioWSRef.current = ws;

    ws.onopen = () => {
      setAudioStatus('connected');
      ws.send(JSON.stringify({ session_id: sessionIdRef.current }));
      startAudioCapture();
    };
    ws.onclose = () => {
      setAudioStatus('disconnected');
      stopAudioCapture();
    };
    ws.onerror = () => {
      setAudioStatus('error');
      stopAudioCapture();
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'ORCHESTRATOR') {
          handleOrchestratorPayload(msg.payload);
        }
      } catch {
        // ignore parse errors
      }
    };

    return () => {
      ws.close();
      stopAudioCapture();
    };
  }, []);

  useEffect(() => {
    // start sending frames when camera is on and embedding WS is connected
    if (cameraStatus !== 'on') {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
      return;
    }

    const sendFrame = () => {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      const ws = embeddingWSRef.current;
      if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

      const W = 640;
      const H = 360;
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, W, H);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
      const b64 = dataUrl.split(',')[1];
      if (!b64) return;

      ws.send(
        JSON.stringify({
          track_id: 'face-1',
          session_id: sessionIdRef.current,
          image_b64: b64,
        })
      );
    };

    sendIntervalRef.current = setInterval(sendFrame, 400);

    return () => {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
    };
  }, [cameraStatus]);

  const stopAudioCapture = () => {
    if (audioProcessorRef.current && audioSourceRef.current) {
      audioSourceRef.current.disconnect();
      audioProcessorRef.current.disconnect();
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((t) => t.stop());
      audioStreamRef.current = null;
    }
    audioSamplesRef.current = [];
  };

  const startAudioCapture = async () => {
    try {
      setAudioError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          sampleSize: 16,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      audioStreamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      audioSourceRef.current = source;
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      audioProcessorRef.current = processor;

      const samplesNeeded = Math.floor(ctx.sampleRate * 0.2); // 200ms

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const buf = audioSamplesRef.current;
        for (let i = 0; i < input.length; i++) {
          buf.push(input[i]);
        }

        while (buf.length >= samplesNeeded) {
          const slice = buf.splice(0, samplesNeeded);
          const pcm = new Int16Array(slice.length);
          for (let i = 0; i < slice.length; i++) {
            const s = Math.max(-1, Math.min(1, slice[i]));
            pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          const ws = audioWSRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(pcm.buffer);
          }
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);
    } catch (e: any) {
      setAudioError(e?.message ?? 'Could not start microphone.');
      setAudioStatus('error');
    }
  };

  return (
    <div className="relative w-full min-h-screen overflow-hidden bg-black">
      {/* Live camera background */}
      <div className="absolute inset-0">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        {/* Subtle overlay tint for readability */}
        <div className="absolute inset-0 bg-slate-900/30 backdrop-blur-[1px]" />
      </div>

      <MetaGlassesOverlay />

      {showPersonCard && (
        <PersonRecognitionCard person={personData} isNewPerson={derivedIsNewPerson} />
      )}

      {showSuggestions && <ConversationSuggestions suggestions={coachFollowups} />}

      {showSummary && (
        <SummarizePanel
          onClose={() => setShowSummary(false)}
          isKnownPerson={!derivedIsNewPerson}
          summaryData={summaryData}
        />
      )}

      {/* Status strip */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 bg-white/5 backdrop-blur-lg border border-white/10 rounded-full px-4 py-2 text-white/80 text-sm">
        <span className="px-2 py-1 rounded-full bg-emerald-400/20 border border-emerald-400/30">
          Session {sessionIdRef.current.slice(0, 8)}
        </span>
        <span className="px-2 py-1 rounded-full bg-white/10">
          Vision WS: {connectionStatus === 'connected' ? 'connected' : connectionStatus}
        </span>
        <span className="px-2 py-1 rounded-full bg-white/10">
          Audio WS: {audioStatus === 'connected' ? 'connected' : audioStatus}
        </span>
        {cameraStatus !== 'on' && (
          <span className="px-2 py-1 rounded-full bg-red-500/20 border border-red-400/30">
            Camera: {cameraError || cameraStatus}
          </span>
        )}
        {audioStatus === 'error' && (
          <span className="px-2 py-1 rounded-full bg-red-500/20 border border-red-400/30">
            Mic: {audioError || 'mic error'}
          </span>
        )}
      </div>

      {/* hidden capture canvas for frame extraction */}
      <canvas ref={captureCanvasRef} className="hidden" />
    </div>
  );
}
