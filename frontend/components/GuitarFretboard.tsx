import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, useTheme } from '@mui/material';

interface FretboardPosition {
  string: number;  // 1-6 (E is 6, e is 1)
  fret: number;    // 0-24 (0 is open string)
  note?: string;   // Optional note name
  duration?: number; // Duration in seconds
  isActive?: boolean; // Whether this position is currently active/highlighted
}

interface GuitarFretboardProps {
  activeFrets: FretboardPosition[];
  highlightScale?: string | null;
  maxFrets?: number;
  showLabels?: boolean;
  showAllNotes?: boolean;
  height?: number | string;
  darkMode?: boolean;
}

// Standard guitar tuning (from low to high)
const STANDARD_TUNING = ['E', 'A', 'D', 'G', 'B', 'E'];

// All possible notes in chromatic order
const ALL_NOTES = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B'];

// Note colors map for visual distinction
const NOTE_COLORS: Record<string, string> = {
  'C': '#e74c3c',
  'C#/Db': '#e67e22',
  'D': '#f1c40f',
  'D#/Eb': '#2ecc71',
  'E': '#1abc9c',
  'F': '#3498db',
  'F#/Gb': '#9b59b6',
  'G': '#8e44ad',
  'G#/Ab': '#2c3e50',
  'A': '#f39c12',
  'A#/Bb': '#d35400',
  'B': '#c0392b',
};

// Function to get note at a specific fret and string
const getNoteAtPosition = (string: number, fret: number): string => {
  // Get the open string note
  const openStringNote = STANDARD_TUNING[string - 1];
  const openStringIndex = ALL_NOTES.findIndex(note => 
    note.includes(openStringNote) && note.length <= 2
  );
  
  // Calculate the note at the given fret
  const noteIndex = (openStringIndex + fret) % 12;
  return ALL_NOTES[noteIndex];
};

// Function to simplify the note name for display
const simplifyNoteName = (note: string): string => {
  if (note.includes('/')) {
    // For accidentals, prefer sharps
    return note.split('/')[0];
  }
  return note;
};

// Function to get the scale notes
const getScaleNotes = (key: string | null): string[] => {
  if (!key) return [];
  
  // Major scale pattern: W-W-H-W-W-W-H (W=whole step, H=half step)
  const majorSteps = [0, 2, 4, 5, 7, 9, 11];
  
  // Find the root note index
  const rootNote = key.includes('m') ? key.replace('m', '') : key;
  const rootIndex = ALL_NOTES.findIndex(note => note.includes(rootNote) && note.length <= 2);
  
  if (rootIndex === -1) return [];
  
  // Generate scale notes
  return majorSteps.map(step => {
    const noteIndex = (rootIndex + step) % 12;
    return ALL_NOTES[noteIndex];
  });
};

const GuitarFretboard: React.FC<GuitarFretboardProps> = ({
  activeFrets = [],
  highlightScale = null,
  maxFrets = 24,
  showLabels = true,
  showAllNotes = false,
  height = '400px',
  darkMode = true
}) => {
  const theme = useTheme();

  // Get scale notes if a key is provided
  const scaleNotes = getScaleNotes(highlightScale);
  
  // Background colors
  const fretboardColor = darkMode ? '#1a1a1a' : '#f5f5dc'; // Dark or light fretboard
  const stringColor = darkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.7)';
  const fretColor = darkMode ? '#333' : '#888';
  const textColor = darkMode ? 'rgba(255, 255, 255, 0.87)' : 'rgba(0, 0, 0, 0.87)';
  const inlayColor = darkMode ? '#333' : '#ddd';
  
  // Calculate fretboard dimensions
  const fretWidth = `calc((100% - 60px) / ${maxFrets})`;
  
  // Inlay positions (standard guitar inlays at frets 3, 5, 7, 9, 12, 15, 17, 19, 21, 24)
  const inlayPositions = [3, 5, 7, 9, 12, 15, 17, 19, 21, 24];

  return (
    <Paper 
      elevation={4} 
      sx={{ 
        p: 2, 
        overflow: 'hidden',
        backgroundColor: darkMode ? '#121212' : '#f7f7f7',
        borderRadius: 2
      }}
    >
      <Typography variant="h5" sx={{ mb: 2, color: textColor, fontWeight: 600 }}>
        Guitar Fretboard {highlightScale ? `(Key: ${highlightScale})` : ''}
      </Typography>
      
      <Box 
        sx={{
          position: 'relative',
          width: '100%',
          height,
          backgroundColor: fretboardColor,
          borderRadius: '8px',
          overflow: 'auto',
          pb: 2
        }}
      >
        {/* Fretboard Container */}
        <Box sx={{ 
          position: 'relative', 
          width: 'fit-content', 
          minWidth: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          padding: '10px 0'
        }}>
          {/* Fret numbers */}
          <Box sx={{ 
            display: 'flex', 
            ml: '60px', // Space for string labels
            borderBottom: `1px solid ${fretColor}`,
            mb: 1,
            pb: 1
          }}>
            {Array.from({ length: maxFrets + 1 }).map((_, fretNum) => (
              <Box 
                key={`fret-label-${fretNum}`}
                sx={{ 
                  width: fretNum === 0 ? '40px' : fretWidth,
                  textAlign: 'center',
                  fontSize: '14px',
                  fontWeight: inlayPositions.includes(fretNum) ? 'bold' : 'normal',
                  color: inlayPositions.includes(fretNum) ? 
                         (darkMode ? theme.palette.primary.main : theme.palette.primary.dark) : 
                         textColor
                }}
              >
                {fretNum}
              </Box>
            ))}
          </Box>
          
          {/* Strings */}
          {Array.from({ length: 6 }).map((_, stringIdx) => {
            const stringNum = stringIdx + 1;
            const stringThickness = 6 - stringIdx; // Thicker for lower strings
            
            return (
              <Box 
                key={`string-${stringNum}`}
                sx={{ 
                  display: 'flex',
                  alignItems: 'center',
                  position: 'relative',
                  height: '40px',
                  '&:not(:last-child)': {
                    borderBottom: `${1 + stringThickness * 0.2}px solid ${stringColor}`
                  }
                }}
              >
                {/* String label */}
                <Box 
                  sx={{ 
                    width: '60px', 
                    textAlign: 'center',
                    fontWeight: 'bold',
                    fontSize: '16px',
                    color: textColor
                  }}
                >
                  {STANDARD_TUNING[stringIdx]}
                </Box>
                
                {/* Frets */}
                {Array.from({ length: maxFrets + 1 }).map((_, fretNum) => {
                  const note = getNoteAtPosition(stringNum, fretNum);
                  const simplifiedNote = simplifyNoteName(note);
                  
                  // Check if this position is active
                  const matchingActive = activeFrets.find(
                    pos => pos.string === stringNum && pos.fret === fretNum
                  );
                  
                  // Check if this note is in the current scale
                  const isInScale = scaleNotes.some(scaleNote => note.includes(scaleNote));
                  
                  // Determine if we should show this note
                  const shouldShowNote = 
                    showAllNotes || 
                    matchingActive || 
                    (highlightScale && isInScale);
                    
                  // Background styling for the nut (fret 0)
                  const isNut = fretNum === 0;
                  
                  // Special styling for fret markers (inlays)
                  const isInlay = inlayPositions.includes(fretNum) && stringNum === 3;
                  
                  // Note color based on the note itself
                  const noteColor = NOTE_COLORS[note.split('/')[0]] || '#ffffff';
                  
                  return (
                    <Box 
                      key={`string-${stringNum}-fret-${fretNum}`}
                      sx={{ 
                        position: 'relative',
                        width: fretNum === 0 ? '40px' : fretWidth,
                        height: '100%',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        borderRight: fretNum > 0 ? `2px solid ${fretColor}` : 'none',
                        backgroundColor: 
                          isNut ? (darkMode ? '#333' : '#ddd') : 
                          'transparent',
                        '&::after': isInlay ? {
                          content: '""',
                          position: 'absolute',
                          width: '14px',
                          height: '14px',
                          borderRadius: '50%',
                          backgroundColor: fretNum === 12 || fretNum === 24 ? 
                                            (darkMode ? theme.palette.primary.main : theme.palette.primary.dark) : 
                                            inlayColor,
                          zIndex: 1
                        } : {}
                      }}
                    >
                      {shouldShowNote && (
                        <Box 
                          sx={{ 
                            width: '30px',
                            height: '30px',
                            borderRadius: '50%',
                            backgroundColor: matchingActive ? 
                              theme.palette.primary.main : 
                              (isInScale ? 'rgba(156, 39, 176, 0.6)' : 'rgba(100, 100, 100, 0.4)'),
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            color: '#fff',
                            fontWeight: 'bold',
                            fontSize: '12px',
                            zIndex: 10,
                            boxShadow: matchingActive ? '0 0 8px rgba(33, 150, 243, 0.6)' : 'none',
                            border: isInScale ? '2px solid rgba(156, 39, 176, 0.8)' : 'none'
                          }}
                        >
                          {simplifiedNote}
                        </Box>
                      )}
                    </Box>
                  );
                })}
              </Box>
            );
          })}
        </Box>
        
        {/* Legend for scale notes */}
        {highlightScale && (
          <Box 
            sx={{ 
              position: 'absolute', 
              bottom: 10, 
              right: 10,
              backgroundColor: 'rgba(0,0,0,0.7)',
              borderRadius: '8px',
              padding: '8px',
              color: '#fff',
              fontSize: '12px'
            }}
          >
            <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>
              {highlightScale} scale
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              {scaleNotes.map((note, i) => (
                <Box 
                  key={`scale-${i}`}
                  sx={{ 
                    padding: '2px 6px',
                    borderRadius: '4px',
                    backgroundColor: 'rgba(156, 39, 176, 0.6)',
                    border: '1px solid rgba(156, 39, 176, 0.8)'
                  }}
                >
                  {simplifyNoteName(note)}
                </Box>
              ))}
            </Box>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default GuitarFretboard;
