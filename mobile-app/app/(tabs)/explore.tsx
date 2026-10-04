import React, { useState, useMemo } from 'react';
import {
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

interface AlphabetEntry {
  letter: string;
  name: string;
  isSupported: boolean;
  isDynamic: boolean;
  summary: string;
  fingers: {
    thumb: string;
    index: string;
    middle: string;
    ring: string;
    pinky: string;
  };
  tips: string;
}

const ISL_ALPHABET_DATA: AlphabetEntry[] = [
  {
    letter: 'A',
    name: 'Letter A',
    isSupported: true,
    isDynamic: false,
    summary: 'Closed fist with thumb resting alongside the index finger.',
    fingers: {
      thumb: 'Upright and resting against side of index',
      index: 'Curled tightly into palm',
      middle: 'Curled tightly into palm',
      ring: 'Curled tightly into palm',
      pinky: 'Curled tightly into palm',
    },
    tips: 'Keep thumb upright along the side of the fist, not folded over the fingers.',
  },
  {
    letter: 'B',
    name: 'Letter B',
    isSupported: true,
    isDynamic: false,
    summary: 'All 4 fingers straight up flat together; thumb tucked across palm.',
    fingers: {
      thumb: 'Tucked across the lower palm',
      index: 'Extended straight up',
      middle: 'Extended straight up (touching index)',
      ring: 'Extended straight up (touching middle)',
      pinky: 'Extended straight up (touching ring)',
    },
    tips: 'Keep all four extended fingers parallel and touching together.',
  },
  {
    letter: 'C',
    name: 'Letter C',
    isSupported: true,
    isDynamic: false,
    summary: 'Hand curved forming a "C" cup shape.',
    fingers: {
      thumb: 'Curved downward forming bottom of "C"',
      index: 'Curved forming top arc of "C"',
      middle: 'Curved with index',
      ring: 'Curved with index',
      pinky: 'Curved with index',
    },
    tips: 'Keep palm facing sideways so the "C" arc is visible to the camera.',
  },
  {
    letter: 'D',
    name: 'Letter D',
    isSupported: true,
    isDynamic: false,
    summary: 'Index finger pointing straight UP; thumb touches middle/ring/pinky.',
    fingers: {
      thumb: 'Touches tips of middle, ring, and pinky in a circle',
      index: 'Pointing straight up vertically',
      middle: 'Curled touching thumb',
      ring: 'Curled touching thumb',
      pinky: 'Curled touching thumb',
    },
    tips: 'Ensure only the index finger is vertical, forming an upright line.',
  },
  {
    letter: 'E',
    name: 'Letter E',
    isSupported: true,
    isDynamic: false,
    summary: 'All 4 fingers curled down tightly touching the thumb tip.',
    fingers: {
      thumb: 'Bent under the fingertips',
      index: 'Curled down touching thumb',
      middle: 'Curled down touching thumb',
      ring: 'Curled down touching thumb',
      pinky: 'Curled down touching thumb',
    },
    tips: 'Curve fingertips inward to rest on the top edge of the thumb.',
  },
  {
    letter: 'F',
    name: 'Letter F',
    isSupported: true,
    isDynamic: false,
    summary: 'Thumb and index form a circle; middle, ring, and pinky stand upright.',
    fingers: {
      thumb: 'Touches index tip in an "OK" circle',
      index: 'Touches thumb tip in an "OK" circle',
      middle: 'Extended straight up and spread',
      ring: 'Extended straight up and spread',
      pinky: 'Extended straight up and spread',
    },
    tips: 'Fan out the 3 upright fingers slightly so MediaPipe detects all three.',
  },
  {
    letter: 'G',
    name: 'Letter G',
    isSupported: true,
    isDynamic: false,
    summary: 'Index and thumb pointing sideways horizontally; others curled.',
    fingers: {
      thumb: 'Extended horizontally parallel to index',
      index: 'Pointing horizontally to the side',
      middle: 'Curled tightly into palm',
      ring: 'Curled tightly into palm',
      pinky: 'Curled tightly into palm',
    },
    tips: 'Point index and thumb sideways across the camera field of view.',
  },
  {
    letter: 'H',
    name: 'Letter H',
    isSupported: false,
    isDynamic: true,
    summary: 'Dynamic motion sign — horizontal index & middle fingers sliding across.',
    fingers: {
      thumb: 'Tucked across lower fingers',
      index: 'Extended horizontally parallel to middle',
      middle: 'Extended horizontally parallel to index',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Currently excluded from real-time static classifier due to motion requirement.',
  },
  {
    letter: 'I',
    name: 'Letter I',
    isSupported: true,
    isDynamic: false,
    summary: 'Only the pinky finger stands straight UP; thumb holds remaining fingers.',
    fingers: {
      thumb: 'Held across index, middle, and ring fingers',
      index: 'Curled into fist',
      middle: 'Curled into fist',
      ring: 'Curled into fist',
      pinky: 'Pointing straight up vertically',
    },
    tips: 'Keep pinky fully extended vertically while keeping other 3 fingers tight in a fist.',
  },
  {
    letter: 'J',
    name: 'Letter J',
    isSupported: false,
    isDynamic: true,
    summary: 'Dynamic motion sign — pinky finger traces a "J" curve in the air.',
    fingers: {
      thumb: 'Holds other fingers into fist',
      index: 'Curled into fist',
      middle: 'Curled into fist',
      ring: 'Curled into fist',
      pinky: 'Extended and tracing a J hook motion',
    },
    tips: 'Motion sign — requires video trajectory tracking (future upgrade).',
  },
  {
    letter: 'K',
    name: 'Letter K',
    isSupported: true,
    isDynamic: false,
    summary: 'Index finger UP, middle finger forward/diagonal, thumb in between.',
    fingers: {
      thumb: 'Rests between index and middle knuckles',
      index: 'Pointing straight up',
      middle: 'Pointing forward / diagonal',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Form a "V" shape and push the middle finger slightly forward with thumb resting between.',
  },
  {
    letter: 'L',
    name: 'Letter L',
    isSupported: true,
    isDynamic: false,
    summary: 'Index pointing straight UP, thumb sticking straight OUT forming an "L".',
    fingers: {
      thumb: 'Extended outward 90° from index',
      index: 'Extended straight UP 90° from thumb',
      middle: 'Curled into palm',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Ensure a clean 90-degree right angle between thumb and index.',
  },
  {
    letter: 'M',
    name: 'Letter M',
    isSupported: true,
    isDynamic: false,
    summary: '3 fingers (Index, Middle, Ring) extended; thumb holds pinky down.',
    fingers: {
      thumb: 'Holds pinky finger down',
      index: 'Extended outward/downward',
      middle: 'Extended outward/downward',
      ring: 'Extended outward/downward',
      pinky: 'Curled and held by thumb',
    },
    tips: 'Show all 3 fingers clearly to the camera.',
  },
  {
    letter: 'N',
    name: 'Letter N',
    isSupported: true,
    isDynamic: false,
    summary: '2 fingers (Index, Middle) extended; thumb holds ring and pinky.',
    fingers: {
      thumb: 'Holds ring and pinky fingers down',
      index: 'Extended outward/downward',
      middle: 'Extended outward/downward',
      ring: 'Curled and held by thumb',
      pinky: 'Curled and held by thumb',
    },
    tips: 'Distinct from M (3 fingers) and V (spread peace sign).',
  },
  {
    letter: 'O',
    name: 'Letter O',
    isSupported: true,
    isDynamic: false,
    summary: 'All 5 fingertips touch to form a complete round circle.',
    fingers: {
      thumb: 'Curves to meet index, middle, ring, pinky tips',
      index: 'Curved touching thumb tip',
      middle: 'Curved touching thumb tip',
      ring: 'Curved touching thumb tip',
      pinky: 'Curved touching thumb tip',
    },
    tips: 'Make sure no fingers stick out so the hand forms a smooth tube/circle.',
  },
  {
    letter: 'P',
    name: 'Letter P',
    isSupported: true,
    isDynamic: false,
    summary: 'Similar to K, but the entire hand is angled downward toward the floor.',
    fingers: {
      thumb: 'Between index and middle',
      index: 'Pointing forward/downward',
      middle: 'Pointing downward',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Angle your wrist downward so the index and middle point toward the floor.',
  },
  {
    letter: 'Q',
    name: 'Letter Q',
    isSupported: true,
    isDynamic: false,
    summary: 'Index and thumb point downward forming a small downward beak/claw.',
    fingers: {
      thumb: 'Pointing downward parallel to index',
      index: 'Pointing downward parallel to thumb',
      middle: 'Curled into palm',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Like a "G" gesture rotated 90° downward.',
  },
  {
    letter: 'R',
    name: 'Letter R',
    isSupported: true,
    isDynamic: false,
    summary: 'Index and middle fingers crossed over each other.',
    fingers: {
      thumb: 'Holds ring and pinky down',
      index: 'Crossed over middle finger',
      middle: 'Crossed behind index finger',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Cross the index finger over the middle finger tightly.',
  },
  {
    letter: 'S',
    name: 'Letter S',
    isSupported: true,
    isDynamic: false,
    summary: 'Tight fist with thumb folded across the front of the fingers.',
    fingers: {
      thumb: 'Wrapped across the front of curled fingers',
      index: 'Curled tightly into fist',
      middle: 'Curled tightly into fist',
      ring: 'Curled tightly into fist',
      pinky: 'Curled tightly into fist',
    },
    tips: 'Thumb goes over the knuckles, distinct from "A" where thumb is along the side.',
  },
  {
    letter: 'T',
    name: 'Letter T',
    isSupported: true,
    isDynamic: false,
    summary: 'Fist with thumb tucked between index and middle fingers.',
    fingers: {
      thumb: 'Protruding between index and middle knuckles',
      index: 'Curled over thumb',
      middle: 'Curled under thumb',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Keep knuckles facing forward to display the tucked thumb.',
  },
  {
    letter: 'U',
    name: 'Letter U',
    isSupported: true,
    isDynamic: false,
    summary: 'Index and middle fingers standing straight UP touching together.',
    fingers: {
      thumb: 'Holds ring and pinky into palm',
      index: 'Extended straight UP touching middle',
      middle: 'Extended straight UP touching index',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Keep index and middle fingers tightly pressed together (unlike V).',
  },
  {
    letter: 'V',
    name: 'Letter V',
    isSupported: true,
    isDynamic: false,
    summary: 'Index and middle fingers spread apart forming a "V" peace sign.',
    fingers: {
      thumb: 'Holds ring and pinky into palm',
      index: 'Extended straight UP angled outward',
      middle: 'Extended straight UP angled outward',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Spread index and middle into an open "V" shape.',
  },
  {
    letter: 'W',
    name: 'Letter W',
    isSupported: true,
    isDynamic: false,
    summary: '3 fingers (Index, Middle, Ring) standing straight UP and spread.',
    fingers: {
      thumb: 'Holds pinky finger into palm',
      index: 'Extended straight UP and spread',
      middle: 'Extended straight UP and spread',
      ring: 'Extended straight UP and spread',
      pinky: 'Curled into palm',
    },
    tips: 'Fan out the three fingers to form a "W" shape.',
  },
  {
    letter: 'X',
    name: 'Letter X',
    isSupported: true,
    isDynamic: false,
    summary: 'Index finger curled into a small hook; other fingers in fist.',
    fingers: {
      thumb: 'Resting against side of fingers',
      index: 'Bent at the knuckle into a hook',
      middle: 'Curled into palm',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Bend only the top joint of the index finger like a pirate hook.',
  },
  {
    letter: 'Y',
    name: 'Letter Y',
    isSupported: false,
    isDynamic: true,
    summary: 'Dynamic motion sign — thumb and pinky extended ("hang loose" gesture).',
    fingers: {
      thumb: 'Extended outward 90°',
      index: 'Curled into palm',
      middle: 'Curled into palm',
      ring: 'Curled into palm',
      pinky: 'Extended outward 90°',
    },
    tips: 'Dynamic sign — excluded from real-time static classifier.',
  },
  {
    letter: 'Z',
    name: 'Letter Z',
    isSupported: true,
    isDynamic: false,
    summary: 'Index finger pointing UP with thumb tucked across lower knuckles.',
    fingers: {
      thumb: 'Tucked across palm base',
      index: 'Pointing upright',
      middle: 'Curled into palm',
      ring: 'Curled into palm',
      pinky: 'Curled into palm',
    },
    tips: 'Hold the pose steady in front of the camera for confirmation.',
  },
];

export default function DictionaryScreen() {
  const [selectedLetter, setSelectedLetter] = useState<string>('A');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterMode, setFilterMode] = useState<'all' | 'ai' | 'dynamic'>('all');

  const filteredList = useMemo(() => {
    return ISL_ALPHABET_DATA.filter((item) => {
      const matchesSearch =
        item.letter.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.tips.toLowerCase().includes(searchQuery.toLowerCase());

      if (!matchesSearch) return false;
      if (filterMode === 'ai') return item.isSupported;
      if (filterMode === 'dynamic') return item.isDynamic;
      return true;
    });
  }, [searchQuery, filterMode]);

  const activeEntry = useMemo(() => {
    return (
      ISL_ALPHABET_DATA.find((item) => item.letter === selectedLetter) ??
      ISL_ALPHABET_DATA[0]
    );
  }, [selectedLetter]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container} bounces={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>📖 ISL Sign Dictionary</Text>
          <Text style={styles.headerSubtitle}>
            Indian Sign Language Alphabet Reference & Pose Guide
          </Text>
        </View>

        {/* Notice Box */}
        <View style={styles.noticeBox}>
          <Text style={styles.noticeText}>
            ℹ️ Text-based anatomical pose descriptions are provided for each letter.
            23 static signs are currently active in the real-time AI classifier.
          </Text>
        </View>

        {/* Search & Filter Bar */}
        <View style={styles.filterSection}>
          <TextInput
            style={styles.searchInput}
            placeholder="Search letters or descriptions..."
            placeholderTextColor="#777"
            value={searchQuery}
            onChangeText={setSearchQuery}
            autoCorrect={false}
          />
          <View style={styles.filterPills}>
            <Pressable
              style={[styles.pill, filterMode === 'all' && styles.pillActive]}
              onPress={() => setFilterMode('all')}
            >
              <Text
                style={[
                  styles.pillText,
                  filterMode === 'all' && styles.pillTextActive,
                ]}
              >
                All (26)
              </Text>
            </Pressable>
            <Pressable
              style={[styles.pill, filterMode === 'ai' && styles.pillActive]}
              onPress={() => setFilterMode('ai')}
            >
              <Text
                style={[
                  styles.pillText,
                  filterMode === 'ai' && styles.pillTextActive,
                ]}
              >
                🟢 AI Active (23)
              </Text>
            </Pressable>
            <Pressable
              style={[styles.pill, filterMode === 'dynamic' && styles.pillActive]}
              onPress={() => setFilterMode('dynamic')}
            >
              <Text
                style={[
                  styles.pillText,
                  filterMode === 'dynamic' && styles.pillTextActive,
                ]}
              >
                ⚠️ Motion Signs (3)
              </Text>
            </Pressable>
          </View>
        </View>

        {/* Alphabet Grid */}
        <Text style={styles.sectionLabel}>Tap an alphabet to view sign pose:</Text>
        <View style={styles.gridContainer}>
          {filteredList.map((item) => {
            const isSelected = item.letter === activeEntry.letter;
            return (
              <Pressable
                key={item.letter}
                style={[
                  styles.gridItem,
                  isSelected && styles.gridItemSelected,
                  !item.isSupported && styles.gridItemDynamic,
                ]}
                onPress={() => setSelectedLetter(item.letter)}
              >
                <Text
                  style={[
                    styles.gridLetter,
                    isSelected && styles.gridLetterSelected,
                    !item.isSupported && styles.gridLetterDynamic,
                  ]}
                >
                  {item.letter}
                </Text>
                <View
                  style={[
                    styles.statusDot,
                    item.isSupported ? styles.dotGreen : styles.dotOrange,
                  ]}
                />
              </Pressable>
            );
          })}
        </View>

        {/* Selected Letter Detail Card */}
        <View style={styles.detailCard}>
          <View style={styles.detailHeader}>
            <View style={styles.bigLetterBadge}>
              <Text style={styles.bigLetterText}>{activeEntry.letter}</Text>
            </View>
            <View style={styles.detailHeaderInfo}>
              <Text style={styles.detailTitle}>{activeEntry.name}</Text>
              <View
                style={[
                  styles.statusBadge,
                  activeEntry.isSupported
                    ? styles.badgeSupported
                    : styles.badgeDynamic,
                ]}
              >
                <Text
                  style={[
                    styles.statusBadgeText,
                    activeEntry.isSupported
                      ? styles.badgeTextSupported
                      : styles.badgeTextDynamic,
                  ]}
                >
                  {activeEntry.isSupported
                    ? '🟢 Real-Time AI Supported'
                    : '⚠️ Motion Sign (Dynamic)'}
                </Text>
              </View>
            </View>
          </View>

          <Text style={styles.summaryText}>{activeEntry.summary}</Text>

          {/* Finger-by-Finger Breakdown */}
          <Text style={styles.breakdownTitle}>✋ Finger-by-Finger Anatomy:</Text>
          <View style={styles.breakdownBox}>
            <View style={styles.fingerRow}>
              <Text style={styles.fingerName}>👍 Thumb:</Text>
              <Text style={styles.fingerDesc}>{activeEntry.fingers.thumb}</Text>
            </View>
            <View style={styles.fingerRow}>
              <Text style={styles.fingerName}>☝️ Index:</Text>
              <Text style={styles.fingerDesc}>{activeEntry.fingers.index}</Text>
            </View>
            <View style={styles.fingerRow}>
              <Text style={styles.fingerName}>🖕 Middle:</Text>
              <Text style={styles.fingerDesc}>{activeEntry.fingers.middle}</Text>
            </View>
            <View style={styles.fingerRow}>
              <Text style={styles.fingerName}>💍 Ring:</Text>
              <Text style={styles.fingerDesc}>{activeEntry.fingers.ring}</Text>
            </View>
            <View style={styles.fingerRow}>
              <Text style={styles.fingerName}>🤙 Pinky:</Text>
              <Text style={styles.fingerDesc}>{activeEntry.fingers.pinky}</Text>
            </View>
          </View>

          {/* Camera Pro Tip */}
          <View style={styles.tipBox}>
            <Text style={styles.tipTitle}>💡 Camera Pro Tip:</Text>
            <Text style={styles.tipText}>{activeEntry.tips}</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  container: {
    padding: 16,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '800',
    color: '#ffffff',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    fontSize: 13,
    color: '#888888',
    marginTop: 4,
  },
  noticeBox: {
    backgroundColor: '#161d26',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#1e3850',
    marginBottom: 16,
  },
  noticeText: {
    color: '#7bc0ff',
    fontSize: 12,
    lineHeight: 17,
  },
  filterSection: {
    marginBottom: 16,
  },
  searchInput: {
    backgroundColor: '#161616',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#262626',
    color: '#ffffff',
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 14,
    marginBottom: 10,
  },
  filterPills: {
    flexDirection: 'row',
    gap: 8,
  },
  pill: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
    backgroundColor: '#1c1c1c',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  pillActive: {
    backgroundColor: '#00ff88',
    borderColor: '#00ff88',
  },
  pillText: {
    color: '#888888',
    fontSize: 12,
    fontWeight: '600',
  },
  pillTextActive: {
    color: '#000000',
    fontWeight: '700',
  },
  sectionLabel: {
    color: '#888888',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 10,
  },
  gridContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 20,
  },
  gridItem: {
    width: 48,
    height: 48,
    borderRadius: 10,
    backgroundColor: '#161616',
    borderWidth: 1.5,
    borderColor: '#262626',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  gridItemSelected: {
    backgroundColor: '#003820',
    borderColor: '#00ff88',
  },
  gridItemDynamic: {
    borderColor: '#3a2a1a',
  },
  gridLetter: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
  },
  gridLetterSelected: {
    color: '#00ff88',
  },
  gridLetterDynamic: {
    color: '#ffaa44',
  },
  statusDot: {
    position: 'absolute',
    top: 4,
    right: 4,
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  dotGreen: {
    backgroundColor: '#00ff88',
  },
  dotOrange: {
    backgroundColor: '#ffaa00',
  },
  detailCard: {
    backgroundColor: '#141414',
    borderRadius: 16,
    padding: 18,
    borderWidth: 1,
    borderColor: '#262626',
  },
  detailHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    marginBottom: 14,
  },
  bigLetterBadge: {
    width: 60,
    height: 60,
    borderRadius: 14,
    backgroundColor: '#00ff88',
    justifyContent: 'center',
    alignItems: 'center',
  },
  bigLetterText: {
    fontSize: 34,
    fontWeight: '900',
    color: '#000000',
  },
  detailHeaderInfo: {
    flex: 1,
  },
  detailTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: '#ffffff',
    marginBottom: 4,
  },
  statusBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 6,
    alignSelf: 'flex-start',
  },
  badgeSupported: {
    backgroundColor: '#003318',
    borderWidth: 1,
    borderColor: '#00ff88',
  },
  badgeDynamic: {
    backgroundColor: '#332000',
    borderWidth: 1,
    borderColor: '#ffaa00',
  },
  statusBadgeText: {
    fontSize: 11,
    fontWeight: '700',
  },
  badgeTextSupported: {
    color: '#00ff88',
  },
  badgeTextDynamic: {
    color: '#ffaa00',
  },
  summaryText: {
    fontSize: 14,
    color: '#cccccc',
    lineHeight: 20,
    marginBottom: 16,
  },
  breakdownTitle: {
    color: '#888888',
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
  },
  breakdownBox: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 12,
    gap: 8,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#222222',
  },
  fingerRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
  },
  fingerName: {
    width: 80,
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '700',
  },
  fingerDesc: {
    flex: 1,
    color: '#bbbbbb',
    fontSize: 12,
    lineHeight: 16,
  },
  tipBox: {
    backgroundColor: '#1b1912',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1,
    borderColor: '#383015',
  },
  tipTitle: {
    color: '#ffcc00',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 4,
  },
  tipText: {
    color: '#ddddbb',
    fontSize: 12,
    lineHeight: 17,
  },
});
