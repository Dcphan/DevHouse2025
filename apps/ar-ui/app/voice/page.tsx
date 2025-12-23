'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
const RECORDING_DURATION_MS = 2600;

type VoiceProfile = {
  id: string;
  name: string;
  created_at: string;
  embedding_length?: number;
};

type RegistrationStatus = 'idle' | 'recording' | 'saving' | 'success' | 'error';
type TestStatus = 'idle' | 'recording' | 'fetching' | 'result' | 'error';
type RecordingMode = 'register' | 'test';

const arrayBufferToBase64 = (buffer: ArrayBuffer) => {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
};

const convertSamplesToBase64 = (samples: number[]) => {
  const int16 = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    int16[i] = value < 0 ? value * 0x8000 : value * 0x7fff;
  }
  return arrayBufferToBase64(int16.buffer);
};

export default function VoiceRegistrationPage() {
  const [profile, setProfile] = useState<VoiceProfile | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [status, setStatus] = useState<RegistrationStatus>('idle');
  const [message, setMessage] = useState<string | null>(null);
  const [testStatus, setTestStatus] = useState<TestStatus>('idle');
  const [testMessage, setTestMessage] = useState<string | null>(null);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [name, setName] = useState('My voice');
  const recordingModeRef = useRef<RecordingMode | null>(null);

  const samplesRef = useRef<number[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const intervalRef = useRef<number | null>(null);
  const startRef = useRef<number | null>(null);
  const nameRef = useRef(name);

  useEffect(() => {
    nameRef.current = name;
  }, [name]);

  const cleanupRecording = useCallback(() => {
    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (gainRef.current) {
      gainRef.current.disconnect();
      gainRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (ctxRef.current) {
      ctxRef.current.close();
      ctxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    samplesRef.current = [];
    startRef.current = null;
    setRecordingTime(0);
  }, []);

  const fetchProfile = useCallback(async () => {
    setLoadingProfile(true);
    try {
      const res = await fetch(`${API_BASE}/voice/profiles`);
      if (!res.ok) throw new Error('Unable to load profile');
      const data = await res.json();
      setProfile(data.profile ?? null);
    } catch (error) {
      console.error(error);
      setProfile(null);
    } finally {
      setLoadingProfile(false);
    }
  }, []);

  const finishRecording = useCallback(async () => {
    const mode = recordingModeRef.current;
    if (!mode) return;
    recordingModeRef.current = null;
    const recorded = [...samplesRef.current];
    cleanupRecording();
    if (recorded.length === 0) {
      if (mode === 'register') {
        setStatus('error');
        setMessage('No audio captured. Try again.');
      } else {
        setTestStatus('error');
        setTestMessage('No audio captured. Try again.');
      }
      return;
    }
    const convertedPayload = {
      audio_base64: convertSamplesToBase64(recorded),
      sample_rate: 16000,
    };
    if (mode === 'register') {
      setStatus('saving');
      setMessage('Saving voice profile...');
      try {
        const response = await fetch(`${API_BASE}/voice/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...convertedPayload, name: nameRef.current }),
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || 'Registration failed');
        }
        const data = await response.json();
        setProfile(data.profile ?? null);
        setStatus('success');
        setMessage('Voice profile updated — the old recording has been replaced.');
      } catch (error: any) {
        console.error(error);
        setStatus('error');
        setMessage(error?.message ?? 'Unable to save voice profile.');
      }
    } else {
      setTestStatus('fetching');
      setTestMessage('Comparing sample to saved voice...');
      setSimilarity(null);
      try {
        const response = await fetch(`${API_BASE}/voice/compare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(convertedPayload),
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || 'Voice comparison failed');
        }
        const data = await response.json();
        const sim = typeof data.similarity === 'number' ? data.similarity : Number(data.similarity);
        const similarityValue = Number.isFinite(sim) ? sim : 0;
        setSimilarity(similarityValue);
        setTestStatus('result');
        setTestMessage(`Similarity: ${(similarityValue * 100).toFixed(1)}%`);
      } catch (error: any) {
        console.error(error);
        setTestStatus('error');
        setTestMessage(error?.message ?? 'Unable to compare voices.');
      }
    }
  }, [cleanupRecording]);

  const startRecording = useCallback(
    async (mode: RecordingMode) => {
      if (recordingModeRef.current) return;
      if (mode === 'register' && (status === 'recording' || status === 'saving')) return;
      if (mode === 'test' && (testStatus === 'recording' || testStatus === 'fetching')) return;
      cleanupRecording();
      if (!navigator.mediaDevices?.getUserMedia) {
        if (mode === 'register') {
          setStatus('error');
          setMessage('Microphone access is not available in this browser.');
        } else {
          setTestStatus('error');
          setTestMessage('Microphone access is not available in this browser.');
        }
        return;
      }
      recordingModeRef.current = mode;
      if (mode === 'register') {
        setStatus('recording');
        setMessage('Recording a 2.5s sample...');
      } else {
        setTestStatus('recording');
        setTestMessage('Recording a sample for comparison...');
        setSimilarity(null);
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: 16000 },
        });
        streamRef.current = stream;
        const ctx = new AudioContext({ sampleRate: 16000 });
        ctxRef.current = ctx;
        const source = ctx.createMediaStreamSource(stream);
        sourceRef.current = source;
        const processor = ctx.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;
        const silenceGain = ctx.createGain();
        silenceGain.gain.value = 0;
        gainRef.current = silenceGain;
        source.connect(processor);
        processor.connect(silenceGain);
        silenceGain.connect(ctx.destination);
        processor.onaudioprocess = (event) => {
          const input = event.inputBuffer.getChannelData(0);
          samplesRef.current.push(...input);
        };
        startRef.current = Date.now();
        intervalRef.current = window.setInterval(() => {
          if (startRef.current) {
            setRecordingTime(Date.now() - startRef.current);
          }
        }, 200);
        timeoutRef.current = window.setTimeout(() => {
          void finishRecording();
        }, RECORDING_DURATION_MS);
      } catch (error: any) {
        console.error(error);
        cleanupRecording();
        recordingModeRef.current = null;
        if (mode === 'register') {
          setStatus('error');
          setMessage(error?.message ?? 'Unable to record audio.');
        } else {
          setTestStatus('error');
          setTestMessage(error?.message ?? 'Unable to record audio.');
        }
      }
    },
    [cleanupRecording, finishRecording, status, testStatus],
  );

  const cancelRecording = useCallback(() => {
    const mode = recordingModeRef.current;
    if (!mode) return;
    recordingModeRef.current = null;
    cleanupRecording();
    if (mode === 'register') {
      setStatus('idle');
      setMessage('Recording canceled.');
    } else {
      setTestStatus('idle');
      setTestMessage('Recording canceled.');
    }
  }, [cleanupRecording]);

  useEffect(() => {
    fetchProfile();
    return () => {
      cleanupRecording();
    };
  }, [cleanupRecording, fetchProfile]);

  const recordingSeconds = useMemo(() => (recordingTime / 1000).toFixed(2), [recordingTime]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-4xl flex-col gap-6 px-4 py-10">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-slate-400">Device setup</p>
            <h1 className="text-3xl font-semibold text-white">Local voice registration</h1>
            <p className="text-sm text-slate-400">
              Capture a short sample on this device—the stored embedding replaces any prior profile.
            </p>
          </div>
          <Link
            href="/"
            className="text-sm text-slate-300 underline-offset-4 transition hover:text-white"
          >
            Back to AR interface
          </Link>
        </header>

        <section className="space-y-3 rounded-2xl border border-white/10 bg-white/5 p-5 shadow-lg shadow-black/20">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Saved voice profile</h2>
            {loadingProfile ? (
              <span className="text-xs text-slate-400">Loading…</span>
            ) : (
              <span className="text-xs text-slate-400">
                {profile ? 'Most recent profile' : 'No profile yet'}
              </span>
            )}
          </div>
          {profile ? (
            <div className="grid gap-2 rounded-xl bg-slate-900/50 p-4 text-sm text-slate-200">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Name</span>
                <span className="font-semibold text-white">{profile.name}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Stored</span>
                <span className="text-slate-200">
                  {new Date(profile.created_at).toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Dimensions</span>
                <span className="text-slate-200">{profile.embedding_length ?? '—'} dims</span>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-slate-700 px-4 py-6 text-sm text-slate-400">
              A voice profile will appear here once you register it.
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-white/10 bg-slate-900/70 p-6 shadow-lg shadow-black/30">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Voice comparison</p>
              <h2 className="text-lg font-semibold text-white">Test a sample</h2>
            </div>
            <span className="text-xs text-slate-400">
              {testStatus === 'recording'
                ? 'Recording…'
                : testStatus === 'fetching'
                  ? 'Comparing…'
                  : testStatus === 'result'
                    ? 'Result ready'
                    : 'Idle'}
            </span>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void startRecording('test')}
              disabled={testStatus === 'recording' || testStatus === 'fetching' || status === 'recording'}
              className="min-w-[200px] rounded-full bg-amber-500 px-5 py-3 text-sm font-semibold text-black transition hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {testStatus === 'recording'
                ? 'Recording…'
                : testStatus === 'fetching'
                  ? 'Comparing…'
                  : 'Record sample to compare'}
            </button>
          </div>
          <div className="mt-4 text-sm text-slate-400">
            {testStatus === 'recording' && `Recording ${recordingSeconds}s`}
            {testStatus === 'fetching' && testMessage}
            {testStatus === 'result' && similarity !== null && (
              <span>
                Similarity vs saved voice: {(similarity * 100).toFixed(1)}%
              </span>
            )}
            {testStatus === 'error' && testMessage}
            {testStatus === 'idle' && !testMessage && 'Record a sample to compare against your saved voice.'}
          </div>
        </section>

        <section className="rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900 to-slate-900/70 p-6 shadow-lg shadow-black/30">
          <div className="mb-4 flex flex-col gap-1">
            <p className="text-sm text-slate-400">Profile label</p>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              className="rounded-lg border border-slate-800 bg-slate-950 px-4 py-3 text-base text-white transition focus:border-slate-500 focus:outline-none"
              placeholder="My voice profile"
            />
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void startRecording('register')}
              disabled={status === 'recording' || status === 'saving' || testStatus === 'recording'}
              className="min-w-[180px] rounded-full bg-emerald-500 px-5 py-3 text-sm font-semibold text-black transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {status === 'recording'
                ? 'Recording…'
                : status === 'saving'
                  ? 'Saving…'
                  : status === 'success'
                    ? 'Saved'
                    : 'Record new voice'}
            </button>
            {status === 'recording' && (
              <>
                <button
                  type="button"
                  onClick={() => void finishRecording()}
                  className="rounded-full border border-slate-600 px-5 py-3 text-sm text-slate-200 transition hover:border-slate-400"
                >
                  Stop & upload
                </button>
                <button
                  type="button"
                  onClick={cancelRecording}
                  className="rounded-full border border-red-500 px-5 py-3 text-sm text-red-300 transition hover:border-red-400"
                >
                  Cancel
                </button>
              </>
            )}
          </div>
          <div className="mt-4 text-sm text-slate-400">
            {status === 'recording' && `Recording ${recordingSeconds}s`}
            {status === 'saving' && 'Sending to the local voice store…'}
            {status === 'success' && message}
            {status === 'error' && message}
            {status === 'idle' && !message && 'Tap record to capture a new sample.'}
            {status === 'idle' && message && message}
          </div>
          <p className="mt-4 text-xs uppercase tracking-[0.3em] text-slate-500">
            Updating your voice replaces the previous stored profile.
          </p>
        </section>
      </div>
    </div>
  );
}
