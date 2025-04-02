import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Box, Typography, Paper, Zoom, Tooltip } from '@mui/material';
import { styled } from '@mui/material/styles';

// Styled components for piano keys
const WhiteKey = styled(Box)(({ theme, active, intensity = 1 }) => ({
  backgroundColor: active ? `rgba(103, 58, 183, ${intensity * 0.7})` : '#fff',
  border: '1px solid #ddd',
  borderRadius: '0 0 6px 6px',
  position: 'relative',
  height: '160px', // Increased height
  flex: 1,
  minWidth: '28px', // Wider keys
  transition: 'all 0.15s ease',
  cursor: 'pointer',
  boxShadow: active 
    ? `0 0 12px rgba(103, 58, 183, ${intensity * 0.8})` 
    : '0 2px 3px rgba(0,0,0,0.1)',
  '&:hover': {
    backgroundColor: active ? `rgba(103, 58, 183, ${intensity * 0.7})` : '#f5f5f5',
  },
  '&:active': {
    transform: 'translateY(2px)',
    boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
  },
  zIndex: 1
}));

const BlackKey = styled(Box)(({ theme, active, intensity = 1 }) => ({
  backgroundColor: active ? `rgba(103, 58, 183, ${intensity * 0.9})` : '#222',
  position: 'absolute',
  width: '60%',
  height: '100px', // Increased height
  top: 0,
  borderRadius: '0 0 5px 5px',
  zIndex: 2,
  transition: 'all 0.15s ease',
  cursor: 'pointer',
  boxShadow: active 
    ? `0 0 12px rgba(103, 58, 183, ${intensity * 0.8})` 
    : '0 3px 6px rgba(0, 0, 0, 0.5)',
  '&:hover': {
    backgroundColor: active ? `rgba(103, 58, 183, ${intensity * 0.9})` : '#333',
  },
  '&:active': {
    transform: 'translateY(2px)',
    boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
  }
}));

const KeyLabel = styled(Typography)(({ active }) => ({
  position: 'absolute',
  bottom: '15px',
  left: 0,
  right: 0,
  textAlign: 'center',
  fontSize: '14px',
  fontWeight: 'bold',
  color: active ? '#5e35b1' : '#888',
  textShadow: active ? '0 0 5px rgba(103, 58, 183, 0.5)' : 'none',
  userSelect: 'none',
  transition: 'all 0.15s ease',
}));

const NoteLabel = styled(Typography)(({ active }) => ({
  position: 'absolute',
  bottom: '35px',
  left: 0,
  right: 0,
  textAlign: 'center',
  fontSize: '12px',
  color: active ? '#512da8' : '#999',
  fontWeight: active ? 'bold' : 'normal',
  userSelect: 'none',
  transition: 'all 0.15s ease',
}));

const PianoKeyboard = ({ notes, currentTime, octaveRange = { min: 2, max: 6 } }) => {
  const keyboardRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [activeNotes, setActiveNotes] = useState([]);
  
  // Piano structure
  const octaves = Array.from({ length: octaveRange.max - octaveRange.min + 1 }, 
    (_, i) => octaveRange.min + i);
  
  const keyMap = useMemo(() => {
    const map = {};
    octaves.forEach(octave => {
      ['C', 'D', 'E', 'F', 'G', 'A', 'B'].forEach((note, index) => {
        map[`${note}${octave}`] = {
          isBlack: false,
          position: (octave - octaveRange.min) * 7 + index,
          midiNote: (octave + 1) * 12 + [0, 2, 4, 5, 7, 9, 11][index]
        };
        
        // Add black keys (sharps/flats)
        if (note !== 'E' && note !== 'B') {
          map[`${note}#${octave}`] = {
            isBlack: true,
            position: (octave - octaveRange.min) * 7 + index,
            offset: 0.65, // Position offset for black keys
            midiNote: (octave + 1) * 12 + [1, 3, null, 6, 8, 10, null][index]
          };
        }
      });
    });
    return map;
  }, [octaves, octaveRange.min]);

  // Calculate dimensions based on container size
  useEffect(() => {
    if (keyboardRef.current) {
      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          setDimensions({
            width: entry.contentRect.width,
            height: entry.contentRect.height
          });
        }
      });
      
      resizeObserver.observe(keyboardRef.current);
      return () => resizeObserver.disconnect();
    }
  }, []);
  
  // Update active notes based on current playback time with enhanced look-ahead
  useEffect(() => {
    if (!notes) return;
    
    const lookAheadTime = 0.2; // Extended look ahead time for better anticipation
    const fadeOutTime = 0.3; // Keep notes visible for 300ms after they end
    
    const active = notes.filter(note => {
      // Calculate the end time with fade out extension
      const effectiveEndTime = note.time + note.duration + fadeOutTime;
      
      // Include notes that are currently playing, about to play, or in fade-out period
      return (currentTime >= note.time - lookAheadTime && 
              currentTime <= effectiveEndTime);
    }).map(note => {
      // Add intensity information for visualization
      let intensity = 1;
      
      // Calculate intensity based on timing
      if (currentTime < note.time) {
        // Note about to play (approaching)
        const approachRatio = 1 - ((note.time - currentTime) / lookAheadTime);
        intensity = 0.5 + (approachRatio * 0.5);
      } else if (currentTime > (note.time + note.duration)) {
        // Note in fade-out period
        const fadeRatio = 1 - ((currentTime - (note.time + note.duration)) / fadeOutTime);
        intensity = fadeRatio;
      }
      
      return {
        ...note,
        intensity: Math.max(0.2, Math.min(1, intensity))
      };
    });
    
    setActiveNotes(active);
  }, [notes, currentTime]);
  
  // Check if a note is active and get its intensity
  const getNoteStatus = (midiNote) => {
    const activeNote = activeNotes.find(note => note.midi === midiNote);
    if (activeNote) {
      return {
        active: true,
        intensity: activeNote.intensity || 1
      };
    }
    return { active: false, intensity: 0 };
  };
  
  // Render piano keys with enhanced styling
  const renderKeys = () => {
    // Render white keys first (as base layer)
    const whiteKeys = Object.entries(keyMap)
      .filter(([_, data]) => !data.isBlack)
      .map(([noteName, data]) => {
        const noteStatus = getNoteStatus(data.midiNote);
        
        return (
          <Tooltip 
            key={noteName} 
            title={`${noteName} (MIDI: ${data.midiNote})`} 
            arrow
            placement="top"
          >
            <WhiteKey active={noteStatus.active} intensity={noteStatus.intensity}>
              <KeyLabel active={noteStatus.active}>{noteName.replace(/\d+$/, '')}</KeyLabel>
              <NoteLabel active={noteStatus.active}>{noteName.match(/\d+$/)}</NoteLabel>
              {noteStatus.active && (
                <Zoom in={noteStatus.active}>
                  <NoteLabel active={noteStatus.active}>
                    {activeNotes.find(note => note.midi === data.midiNote)?.label || ''}
                  </NoteLabel>
                </Zoom>
              )}
            </WhiteKey>
          </Tooltip>
        );
      });
    
    // Then render black keys on top
    const blackKeys = Object.entries(keyMap)
      .filter(([_, data]) => data.isBlack)
      .map(([noteName, data]) => {
        const noteStatus = getNoteStatus(data.midiNote);
        const whiteKeyWidth = dimensions.width / (octaves.length * 7);
        
        return (
          <Tooltip 
            key={noteName} 
            title={`${noteName} (MIDI: ${data.midiNote})`} 
            arrow
            placement="top"
          >
            <BlackKey 
              active={noteStatus.active}
              intensity={noteStatus.intensity}
              sx={{
                left: `calc(${whiteKeyWidth * data.position}px + ${data.offset * whiteKeyWidth}px)`,
                width: `${whiteKeyWidth * 0.65}px` // 65% of white key width
              }}
            >
              <KeyLabel sx={{ color: noteStatus.active ? '#e1bee7' : '#ccc' }} active={noteStatus.active}>
                {noteName.replace(/\d+$/, '')}
              </KeyLabel>
              {noteStatus.active && (
                <Zoom in={noteStatus.active}>
                  <NoteLabel active={noteStatus.active}>
                    {activeNotes.find(note => note.midi === data.midiNote)?.label || ''}
                  </NoteLabel>
                </Zoom>
              )}
            </BlackKey>
          </Tooltip>
        );
      });
    
    return (
      <>
        <Box sx={{ display: 'flex', position: 'relative' }}>
          {whiteKeys}
        </Box>
        <Box sx={{ position: 'relative' }}>
          {blackKeys}
        </Box>
      </>
    );
  };
  
  return (
    <Paper 
      ref={keyboardRef}
      elevation={3} 
      sx={{ 
        p: 3, 
        mb: 3,
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s ease-in-out',
        backgroundColor: '#ffffff',
        borderRadius: '12px',
        border: '1px solid rgba(103, 58, 183, 0.1)',
        boxShadow: '0 8px 24px rgba(103, 58, 183, 0.15)',
        '&:hover': {
          boxShadow: '0 10px 30px rgba(103, 58, 183, 0.2)'
        }
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 500, color: '#512da8' }}>
          Interactive Piano Keyboard
        </Typography>
        
        {activeNotes.length > 0 && (
          <Box sx={{ 
            ml: 2, 
            px: 1.5, 
            py: 0.5, 
            bgcolor: 'rgba(103, 58, 183, 0.1)', 
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center'
          }}>
            <Typography variant="caption" sx={{ fontWeight: 'bold', color: '#512da8' }}>
              {activeNotes.length} Active {activeNotes.length === 1 ? 'Note' : 'Notes'}
            </Typography>
          </Box>
        )}
      </Box>
      <Box sx={{ overflow: 'auto', position: 'relative', height: dimensions.height }}>
        {renderKeys()}
      </Box>
      <Typography variant="caption" sx={{ display: 'block', mt: 1, textAlign: 'center', color: '#666' }}>
        Active notes are highlighted in blue • Scroll horizontally to see more keys
      </Typography>
    </Paper>
  );
};

export default PianoKeyboard;
