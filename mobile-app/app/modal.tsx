import { useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';

import { useSocket, DEFAULT_SERVER_URL } from '@/context/socket-context';

export default function SettingsModalScreen() {
  const router = useRouter();
  const { serverUrl, setServerUrl, reconnect, connected, socket } = useSocket();
  const [inputUrl, setInputUrl] = useState(serverUrl);
  const [saveStatus, setSaveStatus] = useState('');

  const handleSave = () => {
    let clean = inputUrl.trim();
    if (!clean.startsWith('http://') && !clean.startsWith('https://')) {
      clean = 'http://' + clean;
    }
    clean = clean.replace(/\/+$/, '');
    setInputUrl(clean);
    setServerUrl(clean);
    reconnect(clean);
    setSaveStatus('Connecting to ' + clean + '...');
    setTimeout(() => setSaveStatus(''), 3000);
  };

  const handlePreset = (presetUrl: string) => {
    setInputUrl(presetUrl);
    setServerUrl(presetUrl);
    reconnect(presetUrl);
    setSaveStatus('Connecting to ' + presetUrl + '...');
    setTimeout(() => setSaveStatus(''), 3000);
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Status Card */}
        <View style={[styles.statusCard, connected ? styles.cardConnected : styles.cardDisconnected]}>
          <Text style={styles.statusEmoji}>{connected ? '🟢' : '🔴'}</Text>
          <View style={styles.statusInfo}>
            <Text style={styles.statusTitle}>
              {connected ? 'Connected to Backend' : 'Disconnected'}
            </Text>
            <Text style={styles.statusSubtitle} numberOfLines={1}>
              {serverUrl} {socket?.id ? `(ID: ${socket.id.slice(0, 6)}…)` : ''}
            </Text>
          </View>
        </View>

        {/* IP / Server URL Input */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Backend Server URL</Text>
          <Text style={styles.sectionSubtitle}>
            Enter your Mac&apos;s LAN IP address and Flask port (5001).
          </Text>

          <TextInput
            style={styles.input}
            value={inputUrl}
            onChangeText={setInputUrl}
            placeholder="http://192.168.x.x:5001"
            placeholderTextColor="#666"
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />

          <Pressable style={styles.saveBtn} onPress={handleSave}>
            <Text style={styles.saveBtnText}>Save & Reconnect</Text>
          </Pressable>

          {saveStatus ? <Text style={styles.statusAlert}>{saveStatus}</Text> : null}
        </View>

        {/* Quick Presets */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Quick Presets</Text>
          <View style={styles.presetsGrid}>
            <Pressable
              style={styles.presetBtn}
              onPress={() => handlePreset(DEFAULT_SERVER_URL)}
            >
              <Text style={styles.presetBtnTitle}>Default LAN IP</Text>
              <Text style={styles.presetBtnUrl}>{DEFAULT_SERVER_URL}</Text>
            </Pressable>

            <Pressable
              style={styles.presetBtn}
              onPress={() => handlePreset('http://127.0.0.1:5001')}
            >
              <Text style={styles.presetBtnTitle}>iOS Simulator / Localhost</Text>
              <Text style={styles.presetBtnUrl}>http://127.0.0.1:5001</Text>
            </Pressable>

            <Pressable
              style={styles.presetBtn}
              onPress={() => handlePreset('http://10.0.2.2:5001')}
            >
              <Text style={styles.presetBtnTitle}>Android Emulator</Text>
              <Text style={styles.presetBtnUrl}>http://10.0.2.2:5001</Text>
            </Pressable>
          </View>
        </View>

        {/* Tip Box */}
        <View style={styles.tipBox}>
          <Text style={styles.tipTitle}>💡 How to find your Mac&apos;s IP:</Text>
          <Text style={styles.tipText}>
            Open Terminal on your Mac and run:
          </Text>
          <Text style={styles.tipCode}>ipconfig getifaddr en0</Text>
          <Text style={styles.tipText}>
            Ensure your phone and Mac are on the same Wi-Fi network.
          </Text>
        </View>

        {/* Close button */}
        <Pressable style={styles.doneBtn} onPress={() => router.back()}>
          <Text style={styles.doneBtnText}>Done</Text>
        </Pressable>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 36,
  },
  statusCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 20,
  },
  cardConnected: {
    backgroundColor: '#0a2215',
    borderColor: '#00ff88',
  },
  cardDisconnected: {
    backgroundColor: '#220a0a',
    borderColor: '#ff4444',
  },
  statusEmoji: {
    fontSize: 24,
    marginRight: 12,
  },
  statusInfo: {
    flex: 1,
  },
  statusTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: '700',
  },
  statusSubtitle: {
    color: '#aaa',
    fontSize: 12,
    marginTop: 2,
  },
  section: {
    marginBottom: 22,
    backgroundColor: '#121212',
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#222',
  },
  sectionTitle: {
    color: 'white',
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 4,
  },
  sectionSubtitle: {
    color: '#777',
    fontSize: 12,
    marginBottom: 12,
    lineHeight: 16,
  },
  input: {
    backgroundColor: '#1c1c1c',
    color: '#00ff88',
    fontSize: 15,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    marginBottom: 10,
  },
  saveBtn: {
    backgroundColor: '#00ff88',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveBtnText: {
    color: '#000',
    fontSize: 14,
    fontWeight: '700',
  },
  statusAlert: {
    color: '#00ff88',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 8,
  },
  presetsGrid: {
    gap: 8,
  },
  presetBtn: {
    backgroundColor: '#1a1a1a',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  presetBtnTitle: {
    color: '#ddd',
    fontSize: 13,
    fontWeight: '600',
  },
  presetBtnUrl: {
    color: '#666',
    fontSize: 11,
    marginTop: 2,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  tipBox: {
    backgroundColor: '#161616',
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#262626',
    marginBottom: 20,
  },
  tipTitle: {
    color: '#ffaa00',
    fontSize: 13,
    fontWeight: '700',
    marginBottom: 6,
  },
  tipText: {
    color: '#888',
    fontSize: 12,
    lineHeight: 16,
  },
  tipCode: {
    color: '#00ff88',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 13,
    backgroundColor: '#0a0a0a',
    padding: 8,
    borderRadius: 6,
    marginVertical: 6,
  },
  doneBtn: {
    backgroundColor: '#222',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#333',
  },
  doneBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
});
