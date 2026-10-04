import { CameraView, useCameraPermissions } from 'expo-camera';
import { useIsFocused, useRouter } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, Button, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

import { useSocket } from '@/context/socket-context';

const JPEG_QUALITY = 0.3;
const ALLOWED_LETTERS = ['A','B','C','D','E','F','G','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Z'];
const AUTO_INTERVAL_MS = 1000;

export default function CollectScreen() {
  const router = useRouter();
  const { socket, connected, serverUrl } = useSocket();
  const [permission, requestPermission] = useCameraPermissions();
  const isFocused = useIsFocused();
  const cameraRef = useRef<CameraView>(null);
  const isCapturingRef = useRef(false);

  const [selectedLetter, setSelectedLetter] = useState<string>('A');
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState<string>('');
  const [autoEnabled, setAutoEnabled] = useState(false);
  const autoIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [hasCameraError, setHasCameraError] = useState(false);
  const [cameraErrorMsg, setCameraErrorMsg] = useState('');
  const errorCountRef = useRef(0);
  const MAX_CONSECUTIVE_ERRORS = 5;

  // Socket setup — listener subscription on shared socket
  useEffect(() => {
    if (!socket) return;

    const handleCollectResult = (data: any) => {
      console.log('[collect] result', data);
      if (data.status === 'ok') {
        setCounts(data.counts ?? {});
        setTotal(data.total ?? 0);
        setStatus(`Saved ${data.label} (${data.count} total for ${data.label})`);
      } else if (data.status === 'no_hand') {
        setStatus('No hand detected — not saved, try again');
      } else {
        setStatus(`Error: ${data.message ?? 'unknown'}`);
      }
    };

    const handleCollectCounts = (data: any) => {
      if (data.counts) {
        setCounts(data.counts);
        setTotal(data.total ?? 0);
      }
    };

    socket.on('collect_result', handleCollectResult);
    socket.on('collect_counts', handleCollectCounts);

    if (socket.connected) {
      socket.emit('get_collect_counts');
    }

    return () => {
      socket.off('collect_result', handleCollectResult);
      socket.off('collect_counts', handleCollectCounts);
    };
  }, [socket]);

  // Fetch counts when focused
  useEffect(() => {
    if (isFocused && socket?.connected) {
      socket.emit('get_collect_counts');
    }
  }, [isFocused, socket]);

  const restartCamera = useCallback(() => {
    errorCountRef.current = 0;
    setHasCameraError(false);
    setCameraErrorMsg('');
    isCapturingRef.current = false;
    setStatus('Camera recovered — try again');
  }, []);

  const handleCapture = useCallback(async () => {
    if (isCapturingRef.current) return;
    if (!cameraRef.current) {
      setStatus('Camera not ready');
      return;
    }
    if (!isFocused) return;
    if (!socket || !socket.connected) {
      setStatus('Not connected to backend');
      return;
    }
    if (hasCameraError) return;

    isCapturingRef.current = true;
    setStatus(`Capturing ${selectedLetter}...`);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: JPEG_QUALITY,
        shutterSound: false,
        skipProcessing: true,
      });
      if (!photo?.base64) throw new Error('Empty photo');
      // Reset error on success
      errorCountRef.current = 0;
      setHasCameraError(false);
      setCameraErrorMsg('');

      socket.emit('collect_sample', {
        image: photo.base64,
        label: selectedLetter,
      });
      // Don't update counts here; wait for collect_result
    } catch (e: any) {
      const msg = e?.message ?? String(e);
      console.log('[collect] capture error', e);
      errorCountRef.current += 1;
      if (errorCountRef.current >= MAX_CONSECUTIVE_ERRORS) {
        setHasCameraError(true);
        setCameraErrorMsg(msg.includes('could not be captured') ? 'Image could not be captured' : msg);
      } else {
        setStatus(`Capture failed: ${msg} — try again`);
      }
    } finally {
      isCapturingRef.current = false;
    }
  }, [isFocused, selectedLetter, hasCameraError, socket]);

  // Auto-capture loop — reuse safe interval pattern, but 1s not 200ms
  useEffect(() => {
    if (autoEnabled && isFocused && !hasCameraError) {
      console.log('[collect] auto-capture ON');
      autoIntervalRef.current = setInterval(() => {
        handleCapture();
      }, AUTO_INTERVAL_MS);
    } else {
      if (autoIntervalRef.current) {
        clearInterval(autoIntervalRef.current);
        autoIntervalRef.current = null;
      }
    }
    return () => {
      if (autoIntervalRef.current) {
        clearInterval(autoIntervalRef.current);
        autoIntervalRef.current = null;
      }
    };
  }, [autoEnabled, isFocused, hasCameraError, handleCapture]);

  // AppState pause/resume — reuse stability fix
  useEffect(() => {
    const sub = AppState.addEventListener('change', (nextState) => {
      if (nextState !== 'active') {
        isCapturingRef.current = false;
        if (autoIntervalRef.current) {
          clearInterval(autoIntervalRef.current);
          autoIntervalRef.current = null;
        }
      }
    });
    return () => sub.remove();
  }, []);

  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to use the camera</Text>
        <Button onPress={requestPermission} title="Grant permission" />
      </View>
    );
  }

  const selectedCount = counts[selectedLetter] ?? 0;

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="front" />

      {/* Letter selector — scrollable row */}
      <View style={styles.selectorContainer}>
        <Text style={styles.selectorLabel}>Recording letter: {selectedLetter}  •  {selectedCount} samples</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.letterRow}>
          {ALLOWED_LETTERS.map((letter) => {
            const isSelected = letter === selectedLetter;
            const cnt = counts[letter] ?? 0;
            return (
              <TouchableOpacity
                key={letter}
                onPress={() => setSelectedLetter(letter)}
                style={[styles.letterChip, isSelected && styles.letterChipSelected]}
              >
                <Text style={[styles.letterChipText, isSelected && styles.letterChipTextSelected]}>{letter}</Text>
                <Text style={[styles.letterChipCount, isSelected && styles.letterChipCountSelected]}>{cnt}</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
        <Text style={styles.totalText}>Total: {total} samples across {ALLOWED_LETTERS.length} letters</Text>
      </View>

      <View style={styles.overlay}>
        {/* Status / per-letter counter */}
        <TouchableOpacity style={styles.statusBox} onPress={() => router.push('/modal')}>
          <Text style={styles.statusLine}>
            {connected ? '🟢 Connected' : '🔴 Not connected'} ({serverUrl.replace(/^https?:\/\//, '')}) · [Tap for Settings]
          </Text>
          <Text style={styles.countLine}>{selectedLetter}: {selectedCount} samples {selectedCount >= 40 && selectedCount <= 60 ? '✓ good' : selectedCount < 40 ? '(aim 40-60)' : '(enough)'}</Text>
          {status ? <Text style={styles.statusMsg} numberOfLines={2}>{status}</Text> : null}
        </TouchableOpacity>

        {hasCameraError ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorTitle}>Camera error</Text>
            <Text style={styles.errorMsg}>{cameraErrorMsg || 'Image could not be captured'}</Text>
            <Button title="Tap to restart camera" onPress={restartCamera} />
          </View>
        ) : null}

        {/* Capture controls */}
        <View style={styles.captureControls}>
          <View style={styles.captureBtn}><Button title={`Capture ${selectedLetter} Sample`} onPress={handleCapture} disabled={hasCameraError} /></View>
          <View style={styles.captureBtn}>
            <Button
              title={autoEnabled ? 'Stop Auto (1/s)' : 'Start Auto (1/s)'}
              onPress={() => setAutoEnabled((v) => !v)}
              color={autoEnabled ? '#ff4444' : undefined}
              disabled={hasCameraError}
            />
          </View>
        </View>

        {/* Progress hint */}
        <Text style={styles.hint}>Collect 40-60 per letter. Vary angle/distance/lighting. Auto captures every 1s when on.</Text>

        {/* Reliable strip - keep for reference but muted */}
        <Text style={styles.reliableStrip}>Best results: A C E F G L S T U V Z</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center' },
  message: { textAlign: 'center', paddingBottom: 10 },
  camera: { flex: 1 },
  selectorContainer: { backgroundColor: 'black', paddingTop: 8, paddingBottom: 6, borderBottomWidth: 1, borderBottomColor: '#222' },
  selectorLabel: { color: '#00ff88', fontSize: 13, fontWeight: 'bold', textAlign: 'center', marginBottom: 6 },
  letterRow: { paddingHorizontal: 8, gap: 6, flexDirection: 'row' },
  letterChip: { width: 44, height: 48, borderRadius: 8, backgroundColor: '#1a1a1a', borderWidth: 1, borderColor: '#333', alignItems: 'center', justifyContent: 'center' },
  letterChipSelected: { backgroundColor: '#00ff88', borderColor: '#00ff88' },
  letterChipText: { color: 'white', fontSize: 16, fontWeight: 'bold' },
  letterChipTextSelected: { color: 'black' },
  letterChipCount: { color: '#aaa', fontSize: 10, marginTop: 2 },
  letterChipCountSelected: { color: '#222' },
  totalText: { color: '#666', fontSize: 10, textAlign: 'center', marginTop: 6 },
  overlay: { padding: 12, backgroundColor: 'black' },
  statusBox: { padding: 8, backgroundColor: '#111', borderRadius: 8, borderWidth: 1, borderColor: '#222', marginBottom: 8 },
  statusLine: { color: '#aaa', fontSize: 11, textAlign: 'center' },
  countLine: { color: 'white', fontSize: 14, fontWeight: 'bold', textAlign: 'center', marginTop: 4 },
  statusMsg: { color: '#ffaa00', fontSize: 11, textAlign: 'center', marginTop: 4 },
  captureControls: { flexDirection: 'row', gap: 10, marginTop: 8 },
  captureBtn: { flex: 1 },
  hint: { color: '#666', fontSize: 10, textAlign: 'center', marginTop: 8, lineHeight: 14 },
  reliableStrip: { color: '#444', fontSize: 9, textAlign: 'center', marginTop: 6, letterSpacing: 0.5 },
  errorBox: { marginTop: 8, padding: 10, backgroundColor: '#2a1212', borderRadius: 8, borderWidth: 1, borderColor: '#ff4444' },
  errorTitle: { color: '#ff4444', fontWeight: 'bold', fontSize: 12, textAlign: 'center' },
  errorMsg: { color: '#ff9999', fontSize: 10, textAlign: 'center', marginTop: 4, marginBottom: 6 },
});
