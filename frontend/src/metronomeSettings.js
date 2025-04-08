export const metronomeSettings = {
  sounds: [
    { id: 'click', name: 'Click', highFreq: 1000, lowFreq: 800 },
    { id: 'wood', name: 'Wood Block', highFreq: 2400, lowFreq: 1800 },
    { id: 'beep', name: 'Electronic', highFreq: 1500, lowFreq: 1200 }
  ],
  timeSignatures: [
    { id: '4/4', beats: 4, subdivision: 4 },
    { id: '3/4', beats: 3, subdivision: 4 },
    { id: '6/8', beats: 6, subdivision: 8 },
    { id: '2/4', beats: 2, subdivision: 4 }
  ],
  subdivisions: [
    { id: 'quarter', name: 'Quarter Notes', value: 1 },
    { id: 'eighth', name: 'Eighth Notes', value: 2 },
    { id: 'triplet', name: 'Triplets', value: 3 },
    { id: 'sixteenth', name: 'Sixteenth Notes', value: 4 }
  ],
  practiceMode: {
    increments: [1, 2, 3, 4, 5, 8, 10],
    intervals: [1, 2, 4, 8, 16]
  }
};
