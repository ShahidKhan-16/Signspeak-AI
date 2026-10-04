import { CameraView, useCameraPermissions } from 'expo-camera';
import { useIsFocused, useRouter } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AppState, Button, Dimensions, Pressable, StyleSheet, Text, View } from 'react-native';
import * as Speech from 'expo-speech';

import { useSocket } from '@/context/socket-context';

const FRAME_INTERVAL_MS = 200; // 5 fps
const JPEG_QUALITY = 0.3;

// --- Common English words for smart suggestions ---
const WORD_BANK = [
  'HELLO', 'HELP', 'PLEASE', 'THANK', 'THANKS', 'GOOD', 'GREAT', 'FINE',
  'WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW', 'WHICH', 'WILL',
  'YES', 'NO', 'NOT', 'STOP', 'COME', 'LOOK', 'WANT', 'FEEL',
  'LOVE', 'LIKE', 'LIVE', 'GIVE', 'CALL', 'TELL', 'TALK', 'WALK',
  'FOOD', 'WATER', 'HOME', 'WORK', 'SCHOOL', 'TIME', 'OPEN', 'CLOSE',
  'NAME', 'NICE', 'NEED', 'KNOW', 'KEEP', 'KIND', 'LEFT', 'LIST',
  'MORE', 'MEET', 'MAKE', 'MOVE', 'MUST', 'MUCH', 'MIND', 'MISS',
  'OVER', 'ONLY', 'ORDER', 'OTHER', 'QUICK', 'QUIET', 'QUITE',
  'REAL', 'READ', 'REST', 'SAFE', 'SAME', 'SIGN', 'SOME', 'SURE',
  'TRUE', 'TURN', 'UNDER', 'UPON', 'VERY', 'VIEW', 'VOTE', 'WELL',
  'WITH', 'WAIT', 'WISH', 'WORD', 'WORLD', 'WRITE', 'ZERO',
];

// --- Hand landmark mesh connections for drawing skeleton ---
const HAND_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],       // Thumb
  [0,5],[5,6],[6,7],[7,8],       // Index
  [5,9],[9,10],[10,11],[11,12],  // Middle
  [9,13],[13,14],[14,15],[15,16],// Ring
  [13,17],[17,18],[18,19],[19,20],// Pinky
  [0,17],                        // Palm base
];

const FINGER_COLORS: Record<string, string> = {
  thumb: '#FF6B6B',   // Red
  index: '#4ECDC4',   // Teal
  middle: '#FFE66D',  // Yellow
  ring: '#A8E6CF',    // Mint
  pinky: '#DDA0DD',   // Plum
  palm: '#888888',    // Gray
};

function getConnectionColor(a: number, b: number): string {
  if ((a <= 4 && b <= 4)) return FINGER_COLORS.thumb;
  if ((a >= 5 && a <= 8) || (b >= 5 && b <= 8)) {
    if (a === 0 || b === 0 || a === 5 || b === 5) return FINGER_COLORS.index;
    return FINGER_COLORS.index;
  }
  if ((a >= 9 && a <= 12) || (b >= 9 && b <= 12)) return FINGER_COLORS.middle;
  if ((a >= 13 && a <= 16) || (b >= 13 && b <= 16)) return FINGER_COLORS.ring;
  if ((a >= 17 && a <= 20) || (b >= 17 && b <= 20)) return FINGER_COLORS.pinky;
  return FINGER_COLORS.palm;
}

export default function CameraTabScreen() {
  const router = useRouter();
  const { socket, connected, serverUrl } = useSocket();
  const [permission, requestPermission] = useCameraPermissions();
  const isFocused = useIsFocused();
  const cameraRef = useRef<CameraView>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isCapturingRef = useRef(false);

  const [streaming, setStreaming] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const [acksReceived, setAcksReceived] = useState(0);
  const [lastAck, setLastAck] = useState('');
  const [handsDetected, setHandsDetected] = useState<number | null>(null);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [recognizedLetter, setRecognizedLetter] = useState<string>('');
  const [letterConf, setLetterConf] = useState<number | null>(null);
  const CONF_THRESHOLD = 0.45;
  const HOLD_MS = 500;
  const [word, setWord] = useState('');
  const [holdProgress, setHoldProgress] = useState(0);
  const [lastCommittedLetter, setLastCommittedLetter] = useState<string | null>(null);
  const holdRef = useRef<{ letter: string; since: number; missCount: number } | null>(null);
  const lastCommittedRef = useRef<string | null>(null);
  const holdTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Landmark overlay state
  const [landmarks, setLandmarks] = useState<{ x: number; y: number; z: number }[][]>([]);
  const [showLandmarks, setShowLandmarks] = useState(true);

  // TTS state
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Capture error recovery
  const [hasCameraError, setHasCameraError] = useState(false);
  const [cameraErrorMsg, setCameraErrorMsg] = useState('');
  const errorCountRef = useRef(0);
  const skipCountRef = useRef(0);
  const MAX_CONSECUTIVE_ERRORS = 8;
  const BACKOFF_SKIP_COUNT = 4;

  // --- Smart Word Suggestions ---
  const suggestions = useMemo(() => {
    if (!word || word.length < 1) return [];
    const prefix = word.toUpperCase();
    return WORD_BANK.filter((w) => w.startsWith(prefix) && w !== prefix).slice(0, 4);
  }, [word]);

  // --- TTS: Speak word ---
  const speakWord = useCallback((text: string) => {
    if (!ttsEnabled || !text) return;
    setIsSpeaking(true);
    Speech.speak(text, {
      language: 'en-US',
      rate: 0.85,
      pitch: 1.0,
      onDone: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });
  }, [ttsEnabled]);

  const stopSpeech = useCallback(() => {
    Speech.stop();
    setIsSpeaking(false);
  }, []);

  const stopStreaming = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    isCapturingRef.current = false;
  }, []);

  const restartStreaming = useCallback(() => {
    errorCountRef.current = 0;
    skipCountRef.current = 0;
    setHasCameraError(false);
    setCameraErrorMsg('');
    isCapturingRef.current = false;
    setStreaming(true);
  }, []);

  useEffect(() => {
    if (!socket) return;

    const handleFrameAck = (data: any) => {
      setAcksReceived(data.count ?? 0);
      const hands = data.hands_detected ?? null;
      const ms = data.inference_ms ?? null;
      const dropped = data.dropped ? ' (dropped)' : '';
      const handsStr = hands !== null ? ` hands:${hands}` : '';
      const msStr = ms !== null ? ` ${ms}ms` : '';
      setHandsDetected(hands);
      setInferenceMs(ms);
      setLastAck(`ack #${data.count}${handsStr}${msStr}${dropped} (${data.size ? (data.size / 1024).toFixed(1) + 'KB' : data.status})`);

      // Update landmarks for overlay
      if (data.landmarks && Array.isArray(data.landmarks) && data.landmarks.length > 0) {
        setLandmarks(data.landmarks);
      } else {
        setLandmarks([]);
      }
    };

    const clearHold = () => {
      holdRef.current = null;
      setHoldProgress(0);
      if (holdTimerRef.current) { clearInterval(holdTimerRef.current); holdTimerRef.current = null; }
    };
    const startHoldTick = () => {
      if (holdTimerRef.current) clearInterval(holdTimerRef.current);
      holdTimerRef.current = setInterval(() => {
        const h = holdRef.current;
        if (!h) { setHoldProgress(0); return; }
        setHoldProgress(Math.min((Date.now() - h.since) / HOLD_MS, 1));
      }, 50);
    };

    const handleRecognizedLetter = (data: any) => {
      if (data.letter == null) {
        setRecognizedLetter('');
        setLetterConf(null);
        clearHold();
        lastCommittedRef.current = null;
        setLastCommittedLetter(null);
        return;
      }
      const letter: string = data.letter ?? '';
      const conf: number | null = data.confidence ?? null;
      setRecognizedLetter(letter);
      setLetterConf(conf);

      if (!letter || conf == null || conf < CONF_THRESHOLD) {
        if (holdRef.current) {
          holdRef.current.missCount += 1;
          // Allow 2 grace frames before resetting hold
          if (holdRef.current.missCount > 2) {
            clearHold();
            lastCommittedRef.current = null;
            setLastCommittedLetter(null);
          }
        } else {
          lastCommittedRef.current = null;
          setLastCommittedLetter(null);
        }
        return;
      }

      // If this letter was just committed in the current continuous pose,
      // wait until the sign is released or changed before starting a new hold.
      if (lastCommittedRef.current === letter) {
        return;
      }

      const now = Date.now();
      if (!holdRef.current || holdRef.current.letter !== letter) {
        lastCommittedRef.current = null;
        setLastCommittedLetter(null);
        holdRef.current = { letter, since: now, missCount: 0 };
        setHoldProgress(0);
        startHoldTick();
        return;
      }

      // Reset miss count on good frame
      holdRef.current.missCount = 0;
      const elapsed = now - holdRef.current.since;
      setHoldProgress(Math.min(elapsed / HOLD_MS, 1));
      if (elapsed >= HOLD_MS) {
        setWord((prev) => prev + letter);
        lastCommittedRef.current = letter;
        setLastCommittedLetter(letter);
        clearHold();
      }
    };

    socket.on('frame_ack', handleFrameAck);
    socket.on('recognized_letter', handleRecognizedLetter);

    return () => {
      if (holdTimerRef.current) clearInterval(holdTimerRef.current);
      socket.off('frame_ack', handleFrameAck);
      socket.off('recognized_letter', handleRecognizedLetter);
    };
  }, [socket]);

  const captureAndSendFrame = useCallback(async () => {
    if (skipCountRef.current > 0) {
      skipCountRef.current -= 1;
      return;
    }
    if (hasCameraError) return;
    if (isCapturingRef.current) return;
    if (!cameraRef.current) return;
    if (!isFocused || !streaming) return;
    if (!socket || !socket.connected) return;

    isCapturingRef.current = true;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: JPEG_QUALITY,
        shutterSound: false,
        skipProcessing: true,
      });
      if (photo?.base64) {
        if (errorCountRef.current !== 0) console.log('[capture] recovered after', errorCountRef.current, 'failures');
        errorCountRef.current = 0;
        if (hasCameraError) {
          setHasCameraError(false);
          setCameraErrorMsg('');
        }
        skipCountRef.current = 0;
        socket.emit('video_frame', {
          image: photo.base64,
          timestamp: Date.now(),
          width: photo.width,
          height: photo.height,
        });
        setFramesSent((c) => c + 1);
      } else {
        throw new Error('Empty photo result');
      }
    } catch (e: any) {
      const msg = e?.message ?? String(e);
      console.log('capture error', e);
      errorCountRef.current += 1;
      skipCountRef.current = BACKOFF_SKIP_COUNT;
      if (errorCountRef.current >= MAX_CONSECUTIVE_ERRORS) {
        console.log(`[capture] threshold reached (${errorCountRef.current}) — stopping`);
        setHasCameraError(true);
        setCameraErrorMsg(msg.includes('could not be captured') ? 'Image could not be captured' : msg);
        stopStreaming();
      }
    } finally {
      isCapturingRef.current = false;
    }
  }, [isFocused, streaming, hasCameraError, stopStreaming, socket]);

  const startStreaming = useCallback(() => {
    if (intervalRef.current) return;
    if (hasCameraError) {
      console.log('[stream] not starting — camera error, needs restart');
      return;
    }
    captureAndSendFrame();
    intervalRef.current = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
  }, [captureAndSendFrame, hasCameraError]);

  useEffect(() => {
    if (hasCameraError) {
      stopStreaming();
      return () => stopStreaming();
    }
    if (streaming && isFocused && connected) startStreaming();
    else stopStreaming();
    return () => stopStreaming();
  }, [streaming, isFocused, connected, hasCameraError, startStreaming, stopStreaming]);

  useEffect(() => () => {
    stopStreaming();
    if (holdTimerRef.current) clearInterval(holdTimerRef.current);
  }, [stopStreaming]);

  useEffect(() => {
    const sub = AppState.addEventListener('change', (nextState) => {
      if (nextState !== 'active') {
        stopStreaming();
        isCapturingRef.current = false;
        skipCountRef.current = 0;
      } else if (streaming && isFocused && connected && !hasCameraError) {
        setTimeout(() => { if (cameraRef.current) startStreaming(); }, 600);
      }
    });
    return () => sub.remove();
  }, [streaming, isFocused, connected, hasCameraError, startStreaming, stopStreaming]);

  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to use the camera</Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  const screenWidth = Dimensions.get('window').width;
  const cameraHeight = screenWidth * (4 / 3); // Camera aspect ratio

  return (
    <View style={styles.container}>
      {/* Camera + Landmark Overlay */}
      <View style={{ width: screenWidth, height: cameraHeight, position: 'relative' }}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="front" />

        {/* Hand Landmark Mesh Overlay */}
        {showLandmarks && landmarks.length > 0 && (
          <View style={StyleSheet.absoluteFill} pointerEvents="none">
            {landmarks.map((hand, handIdx) => (
              <View key={handIdx} style={StyleSheet.absoluteFill}>
                {/* Draw connection lines */}
                {HAND_CONNECTIONS.map(([a, b], connIdx) => {
                  if (a >= hand.length || b >= hand.length) return null;
                  // Invert X for front camera selfie preview mirroring
                  const x1 = (1 - hand[a].x) * screenWidth;
                  const y1 = hand[a].y * cameraHeight;
                  const x2 = (1 - hand[b].x) * screenWidth;
                  const y2 = hand[b].y * cameraHeight;
                  const dx = x2 - x1;
                  const dy = y2 - y1;
                  const len = Math.sqrt(dx * dx + dy * dy);
                  const angle = Math.atan2(dy, dx) * (180 / Math.PI);
                  const color = getConnectionColor(a, b);
                  return (
                    <View
                      key={`conn-${handIdx}-${connIdx}`}
                      style={{
                        position: 'absolute',
                        left: x1,
                        top: y1,
                        width: len,
                        height: 2.5,
                        backgroundColor: color,
                        opacity: 0.8,
                        transform: [{ rotate: `${angle}deg` }],
                        transformOrigin: 'left center',
                      }}
                    />
                  );
                })}
                {/* Draw landmark dots */}
                {hand.map((lm, lmIdx) => {
                  const isTip = [4, 8, 12, 16, 20].includes(lmIdx);
                  const dotSize = isTip ? 8 : 5;
                  // Invert X for front camera selfie preview mirroring
                  const px = (1 - lm.x) * screenWidth;
                  const py = lm.y * cameraHeight;
                  return (
                    <View
                      key={`dot-${handIdx}-${lmIdx}`}
                      style={{
                        position: 'absolute',
                        left: px - dotSize / 2,
                        top: py - dotSize / 2,
                        width: dotSize,
                        height: dotSize,
                        borderRadius: dotSize / 2,
                        backgroundColor: isTip ? '#ffffff' : '#00ff88',
                        borderWidth: isTip ? 1.5 : 0,
                        borderColor: '#000',
                        opacity: 0.9,
                      }}
                    />
                  );
                })}
              </View>
            ))}
          </View>
        )}

        {/* Toggle buttons on camera */}
        <View style={styles.cameraControls}>
          <Pressable
            style={[styles.camBtn, showLandmarks && styles.camBtnActive]}
            onPress={() => setShowLandmarks((v) => !v)}
          >
            <Text style={styles.camBtnText}>{showLandmarks ? '🦴' : '🦴'}</Text>
          </Pressable>
          <Pressable
            style={[styles.camBtn, ttsEnabled && styles.camBtnActive]}
            onPress={() => { setTtsEnabled((v) => !v); if (isSpeaking) stopSpeech(); }}
          >
            <Text style={styles.camBtnText}>{ttsEnabled ? '🔊' : '🔇'}</Text>
          </Pressable>
          <Pressable
            style={styles.camBtn}
            onPress={() => router.push('/modal')}
          >
            <Text style={styles.camBtnText}>⚙️</Text>
          </Pressable>
        </View>
      </View>

      {/* Bottom UI Panel */}
      <View style={styles.overlay}>
        {/* Word Builder */}
        <View style={styles.wordContainer}>
          <Text style={styles.wordLabel}>Word so far</Text>
          <Text style={styles.wordBuilt} numberOfLines={2}>{word ? word.split('').join(' ') : '—'}</Text>
          <View style={styles.wordControls}>
            <Pressable
              style={styles.actionBtn}
              onPress={() => {
                setWord((p) => p.slice(0, -1));
                lastCommittedRef.current = null;
                setLastCommittedLetter(null);
              }}
            >
              <Text style={styles.actionBtnText}>⌫ Backspace</Text>
            </Pressable>
            <Pressable
              style={styles.actionBtn}
              onPress={() => {
                setWord('');
                holdRef.current = null;
                lastCommittedRef.current = null;
                setLastCommittedLetter(null);
                setHoldProgress(0);
              }}
            >
              <Text style={[styles.actionBtnText, { color: '#ff4444' }]}>Reset</Text>
            </Pressable>
            <Pressable
              style={[styles.actionBtn, styles.speakBtn, isSpeaking && styles.speakBtnActive]}
              onPress={() => { if (isSpeaking) stopSpeech(); else speakWord(word); }}
              disabled={!word}
            >
              <Text style={[styles.actionBtnText, { color: '#4ECDC4' }]}>
                {isSpeaking ? '⏹ Stop' : '🔊 Speak'}
              </Text>
            </Pressable>
          </View>
        </View>

        {/* Smart Word Suggestions */}
        {suggestions.length > 0 && (
          <View style={styles.suggestionsRow}>
            {suggestions.map((s) => (
              <Pressable
                key={s}
                style={styles.suggestionChip}
                onPress={() => {
                  setWord(s);
                  lastCommittedRef.current = null;
                  setLastCommittedLetter(null);
                  if (ttsEnabled) speakWord(s);
                }}
              >
                <Text style={styles.suggestionText}>{s}</Text>
              </Pressable>
            ))}
          </View>
        )}

        {/* Live Letter Recognition */}
        {recognizedLetter ? (
          <View style={[styles.letterBox, holdProgress > 0 && holdProgress < 1 && styles.letterBoxHolding]}>
            <View style={styles.liveRow}>
              <Text style={styles.liveLabel}>Live</Text>
              <Text style={[styles.letter, holdProgress > 0 && styles.letterPulsing]}>{recognizedLetter}</Text>
              <Text style={[styles.letterConf, (letterConf ?? 0) >= CONF_THRESHOLD ? styles.confGreen : (letterConf ?? 0) >= 0.4 ? styles.confOrange : styles.confRed]}>
                {letterConf !== null ? `${Math.round(letterConf * 100)}%` : ''}
              </Text>
            </View>
            <View style={styles.confBarTrack}>
              <View style={[styles.confBarFill, { width: `${Math.min(100, Math.round(((letterConf ?? 0) * 100)))}%` }, (letterConf ?? 0) >= CONF_THRESHOLD ? styles.confBarGreen : (letterConf ?? 0) >= 0.4 ? styles.confBarOrange : styles.confBarRed]} />
              <View style={[styles.confThreshold, { left: `${CONF_THRESHOLD * 100}%` }]} />
            </View>
            {(letterConf ?? 0) >= CONF_THRESHOLD && holdProgress > 0 && holdProgress < 1 ? (
              <View style={styles.holdBarTrack}><View style={[styles.holdBarFill, { width: `${Math.round(holdProgress * 100)}%` }]} /></View>
            ) : null}
            <Text style={styles.holdHint}>
              {(letterConf ?? 0) < CONF_THRESHOLD
                ? `Need ${Math.round(CONF_THRESHOLD * 100)}% to lock — hold steady`
                : lastCommittedLetter === recognizedLetter
                ? 'Letter added ✓ Release sign to repeat'
                : holdProgress > 0 && holdProgress < 1
                ? `Holding… ${Math.round(holdProgress * 100)}%`
                : 'Hold steady to lock'}
            </Text>
          </View>
        ) : (
          <Text style={styles.letterHint}>{hasCameraError ? 'Camera paused' : handsDetected ? 'Hold a sign steady…' : 'Show an ISL letter (A-Z)'}</Text>
        )}

        {/* Camera Error */}
        {hasCameraError ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorTitle}>Camera error</Text>
            <Text style={styles.errorMsg} numberOfLines={2}>{cameraErrorMsg || 'Image could not be captured'}</Text>
            <Text style={styles.errorHint}>Streaming stopped. Tap to restart.</Text>
            <Button title="Tap to restart streaming" onPress={restartStreaming} />
          </View>
        ) : null}

        {/* Stream Control */}
        <Pressable
          style={[styles.streamBtn, streaming && styles.streamBtnStop]}
          onPress={() => setStreaming((s) => !s)}
        >
          <Text style={styles.streamBtnText}>{streaming ? '⏹ Stop streaming' : '▶ Start streaming'}</Text>
        </Pressable>
        {!isFocused && streaming ? <Text style={styles.hint}>Paused — tab not focused</Text> : null}

        {/* Debug Info */}
        <Pressable style={styles.debugBox} onPress={() => router.push('/modal')}>
          <Text style={styles.debugText}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'} ({serverUrl.replace(/^https?:\/\//, '')}) · {streaming && isFocused && connected ? `Streaming ${1000 / FRAME_INTERVAL_MS}fps (sent ${framesSent} · ack ${acksReceived})` : hasCameraError ? 'Error — tap restart' : 'Idle'} {handsDetected !== null ? `· Hands ${handsDetected} ${inferenceMs ? `· ${inferenceMs}ms` : ''}` : ''} · [Tap ⚙️ for Settings]
          </Text>
          {lastAck ? <Text style={styles.debugText} numberOfLines={1}>{lastAck}</Text> : null}
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  message: { textAlign: 'center', paddingBottom: 10, color: 'white' },
  overlay: { flex: 1, padding: 12, backgroundColor: '#0a0a0a' },

  // Camera overlay controls
  cameraControls: {
    position: 'absolute', top: 12, right: 12,
    flexDirection: 'column', gap: 8,
  },
  camBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.5)', alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.2)',
  },
  camBtnActive: { borderColor: '#00ff88', backgroundColor: 'rgba(0,255,136,0.15)' },
  camBtnText: { fontSize: 18 },

  // Word builder
  wordContainer: { padding: 12, backgroundColor: '#111', borderRadius: 12, borderWidth: 1, borderColor: '#1a1a1a' },
  wordLabel: { color: '#555', fontSize: 9, letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 4, textAlign: 'center' },
  wordBuilt: { color: 'white', fontSize: 36, fontWeight: '900', letterSpacing: 5, textAlign: 'center', minHeight: 42 },
  wordControls: { flexDirection: 'row', justifyContent: 'center', gap: 8, marginTop: 8 },
  actionBtn: {
    paddingHorizontal: 12, paddingVertical: 6,
    backgroundColor: '#1a1a1a', borderRadius: 8, borderWidth: 1, borderColor: '#2a2a2a',
  },
  actionBtnText: { color: '#aaa', fontSize: 12, fontWeight: '600' },
  speakBtn: { borderColor: '#4ECDC4' },
  speakBtnActive: { backgroundColor: 'rgba(78,205,196,0.15)' },

  // Suggestions
  suggestionsRow: { flexDirection: 'row', gap: 6, marginTop: 8, justifyContent: 'center', flexWrap: 'wrap' },
  suggestionChip: {
    paddingHorizontal: 14, paddingVertical: 6,
    backgroundColor: '#1a1a2e', borderRadius: 16,
    borderWidth: 1, borderColor: '#4ECDC4',
  },
  suggestionText: { color: '#4ECDC4', fontSize: 13, fontWeight: '600', letterSpacing: 0.5 },

  // Live letter recognition
  liveLabel: { color: '#666', fontSize: 9, letterSpacing: 1, textTransform: 'uppercase' },
  letterBox: { alignItems: 'center', marginTop: 8, padding: 8, backgroundColor: '#111', borderRadius: 12, borderWidth: 1, borderColor: '#1a1a1a' },
  letterBoxHolding: { borderColor: '#00ff88', borderWidth: 1.5 },
  liveRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  letter: { color: '#00ff88', fontSize: 32, fontWeight: 'bold', textAlign: 'center' },
  letterPulsing: { color: '#aaffcc' },
  letterConf: { fontSize: 13, fontWeight: '600' },
  confGreen: { color: '#00ff88' },
  confOrange: { color: '#ffaa00' },
  confRed: { color: '#ff4444' },
  letterHint: { color: '#555', fontSize: 11, textAlign: 'center', marginTop: 6, fontStyle: 'italic' },
  confBarTrack: { height: 5, backgroundColor: '#222', borderRadius: 3, overflow: 'hidden', width: '100%', marginTop: 4, position: 'relative' },
  confBarFill: { height: '100%', borderRadius: 3 },
  confBarGreen: { backgroundColor: '#00ff88' },
  confBarOrange: { backgroundColor: '#ffaa00' },
  confBarRed: { backgroundColor: '#ff4444' },
  confThreshold: { position: 'absolute', top: 0, bottom: 0, width: 2, backgroundColor: 'white', opacity: 0.9 },
  holdBarTrack: { height: 3, backgroundColor: '#1a1a1a', borderRadius: 2, overflow: 'hidden', width: '100%', marginTop: 4 },
  holdBarFill: { height: '100%', backgroundColor: '#00ff88', borderRadius: 2 },
  holdHint: { color: '#666', fontSize: 9, textAlign: 'center', marginTop: 3 },

  // Stream button
  streamBtn: {
    marginTop: 8, paddingVertical: 10,
    backgroundColor: '#1a3a2a', borderRadius: 10,
    alignItems: 'center', borderWidth: 1, borderColor: '#00ff88',
  },
  streamBtnStop: { backgroundColor: '#3a1a1a', borderColor: '#ff4444' },
  streamBtnText: { color: 'white', fontSize: 14, fontWeight: '700' },

  hint: { color: '#ffaa00', fontSize: 11, marginTop: 4, textAlign: 'center' },
  errorBox: { marginTop: 8, padding: 10, backgroundColor: '#2a1212', borderRadius: 10, borderWidth: 1, borderColor: '#ff4444' },
  errorTitle: { color: '#ff4444', fontWeight: 'bold', fontSize: 12, textAlign: 'center' },
  errorMsg: { color: '#ff9999', fontSize: 10, textAlign: 'center', marginTop: 3 },
  errorHint: { color: '#aaa', fontSize: 9, textAlign: 'center', marginTop: 4, marginBottom: 6 },
  debugBox: { marginTop: 8, paddingTop: 6, borderTopWidth: 1, borderTopColor: '#1a1a1a', opacity: 0.5 },
  debugText: { color: '#555', fontSize: 9, lineHeight: 12 },
});
