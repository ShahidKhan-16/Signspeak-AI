import CameraScreen from './camera';

/**
 * Main Translate Screen (Home tab).
 * Directly renders the complete Camera / Live Translator implementation
 * without duplicating any camera, socket, or ML logic.
 */
export default function TranslateScreen() {
  return <CameraScreen />;
}
