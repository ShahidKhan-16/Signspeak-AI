import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface SocketContextType {
  socket: Socket | null;
  connected: boolean;
  serverUrl: string;
  setServerUrl: (url: string) => void;
  reconnect: (newUrl?: string) => void;
}

export const DEFAULT_SERVER_URL = 'http://192.168.29.116:5001';

const SocketContext = createContext<SocketContextType>({
  socket: null,
  connected: false,
  serverUrl: DEFAULT_SERVER_URL,
  setServerUrl: () => {},
  reconnect: () => {},
});

export function SocketProvider({ children }: { children: React.ReactNode }) {
  const [serverUrl, setServerUrlState] = useState(DEFAULT_SERVER_URL);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);
  const [, setRerender] = useState(0);

  const connectSocket = (url: string) => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    try {
      const cleanUrl = url.trim().replace(/\/+$/, '');
      console.log('[SocketContext] Connecting to:', cleanUrl);
      const newSocket = io(cleanUrl, {
        transports: ['websocket'],
        reconnection: true,
        reconnectionAttempts: Infinity,
        reconnectionDelay: 1000,
        timeout: 10000,
      });

      newSocket.on('connect', () => {
        console.log('[SocketContext] Connected to backend! ID:', newSocket.id);
        setConnected(true);
      });

      newSocket.on('disconnect', (reason) => {
        console.log('[SocketContext] Disconnected from backend:', reason);
        setConnected(false);
      });

      newSocket.on('connect_error', (error) => {
        console.log('[SocketContext] Connection error:', error.message);
        setConnected(false);
      });

      socketRef.current = newSocket;
      setRerender((v) => v + 1);
    } catch (e) {
      console.error('[SocketContext] Failed to initialize socket:', e);
      setConnected(false);
    }
  };

  useEffect(() => {
    connectSocket(serverUrl);
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [serverUrl]);

  const setServerUrl = (url: string) => {
    const cleanUrl = url.trim().replace(/\/+$/, '');
    setServerUrlState(cleanUrl);
  };

  const reconnect = (newUrl?: string) => {
    const targetUrl = newUrl ? newUrl.trim().replace(/\/+$/, '') : serverUrl;
    if (newUrl) {
      setServerUrlState(targetUrl);
    }
    connectSocket(targetUrl);
  };

  return (
    <SocketContext.Provider
      value={{
        socket: socketRef.current,
        connected,
        serverUrl,
        setServerUrl,
        reconnect,
      }}
    >
      {children}
    </SocketContext.Provider>
  );
}

export function useSocket() {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useSocket must be used within a SocketProvider');
  }
  return context;
}
