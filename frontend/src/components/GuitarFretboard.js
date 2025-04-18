import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Box, Typography, Paper, Tooltip, Chip, Fade } from '@mui/material';
import { styled } from '@mui/material/styles';
import InfoIcon from '@mui/icons-material/Info';

const GuitarFretboard = ({ notes, currentTime, analysisData }) => {
  // Enhanced interactive fretboard visualization
  const fretboardRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [activeNotes, setActiveNotes] = useState([]);
  
  // Standard tuning strings and fret positions
  const strings = ['E', 'B', 'G', 'D', 'A', 'E'];
  const numberOfFrets = 15;
  
  // Calculate dimensions based on container size
  useEffect(() => {
    if (fretboardRef.current) {
      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          const containerWidth = entry.contentRect.width;
          // Set height based on a ratio of the width, but with a minimum height
          setDimensions({
            width: containerWidth,
            height: Math.max(250, containerWidth / 250) // Wider ratio for better visibility
          });
        }
      });
      
      resizeObserver.observe(fretboardRef.current);
      return () => resizeObserver.disconnect();
    }
  }, []);
  
  // Update active notes based on current playback time with enhanced look-ahead
  useEffect(() => {
    if (!notes) return;
    
    const lookAheadTime = 0.2; // Look ahead 200ms to prepare animations
    const fadeOutTime = 0.3; // Keep notes visible for 300ms after they end for better visualization
    
    const active = notes.filter(note => {
      // Calculate the end time with fade out extension
      const effectiveEndTime = note.time + note.duration + fadeOutTime;
      
      // Include notes that are currently playing, about to play or in fade-out period
      return (currentTime >= note.time - lookAheadTime && 
              currentTime <= effectiveEndTime);
    });
    
    setActiveNotes(active);
  }, [notes, currentTime]);
  
  // Memoize note positions for better performance
  const getNotePosition = useMemo(() => {
    return (noteName) => {
      // Standard tuning open strings
      const openStrings = [
        { string: 0, note: 'E4' }, // High E
        { string: 1, note: 'B3' },
        { string: 2, note: 'G3' },
        { string: 3, note: 'D3' },
        { string: 4, note: 'A2' },
        { string: 5, note: 'E2' }  // Low E
      ];
      
      // Function to get pitch class and octave from note name
      const getPitchClass = (note) => {
        // Extract note and octave (e.g., "C#4" -> "C#", 4)
        const match = note.match(/(^[A-G][#b]?)(\d+)$/);
        if (!match) return { pitchClass: note, octave: 4 }; // Default octave if not specified
        
        return {
          pitchClass: match[1],
          octave: parseInt(match[2])
        };
      };
      
      // Calculate semitones between notes
      const semitones = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
      };
      
      const calculateSemitones = (note1, note2) => {
        const n1 = getPitchClass(note1);
        const n2 = getPitchClass(note2);
        
        const st1 = semitones[n1.pitchClass] + (n1.octave * 12);
        const st2 = semitones[n2.pitchClass] + (n2.octave * 12);
        
        return st2 - st1;
      };
      
      // Find possible positions
      const positions = [];
      openStrings.forEach(string => {
        const semitonesFromOpen = calculateSemitones(string.note, noteName);
        if (semitonesFromOpen >= 0 && semitonesFromOpen <= numberOfFrets) {
          positions.push({
            string: string.string,
            fret: semitonesFromOpen
          });
        }
      });
      
      // If found, return first (usually most playable) position
      if (positions.length > 0) {
        // Sort by lower fret numbers first (typically easier to play)
        positions.sort((a, b) => a.fret - b.fret);
        return positions[0];
      }
      
      return null;
    };
  }, [numberOfFrets]);
  
  // Initialize with default dimensions if not yet calculated
  useEffect(() => {
    if (dimensions.width === 0 && fretboardRef.current) {
      const width = fretboardRef.current.clientWidth;
      setDimensions({
        width: width,
        height: Math.max(250, width / 250)
      });
    }
  }, [dimensions.width]);
  
  // Calculate spacing for the fretboard
  const stringSpacing = dimensions.height / (strings.length - 1);
  const fretSpacing = dimensions.width / (numberOfFrets + 1);
  
  return (
    <Paper 
      ref={fretboardRef}
      elevation={3} 
      sx={{ 
        position: 'relative', 
        width: '100%', 
        height: dimensions.height,
        overflow: 'hidden',
        mb: 2,
        backgroundColor: '#f5f5f7',
        borderRadius: '8px',
        border: '1px solid rgba(103, 58, 183, 0.2)',
        boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
        '&:hover': {
          boxShadow: '0 10px 25px rgba(103, 58, 183, 0.25)'
        },
        // Ensure full width on all devices
        maxWidth: '100%',
        mx: 'auto'
      }}
    >
      <Typography variant="h6" gutterBottom>
        Guitar Fretboard Visualization
      </Typography>
      
      <Box 
        sx={{
          position: 'relative',
          width: '100%',
          height: dimensions.height,
          mt: 2
        }}
      >
        {/* Fretboard */}
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundColor: '#D2B48C', // Tan color for fretboard
            borderRadius: '4px',
            boxShadow: 'inset 0 0 10px rgba(0,0,0,0.3)'
          }}
        />
        
        {/* Frets */}
        {[...Array(numberOfFrets + 1)].map((_, fretIndex) => (
          <Box
            key={`fret-${fretIndex}`}
            sx={{
              position: 'absolute',
              top: 0,
              left: fretSpacing * fretIndex,
              width: fretIndex === 0 ? 3 : 2,
              height: '100%',
              backgroundColor: fretIndex === 0 ? '#333' : '#999',
              zIndex: 2,
              boxShadow: fretIndex === 0 ? '1px 0 3px rgba(0,0,0,0.5)' : 'none'
            }}
          />
        ))}
        
        {/* Fret markers (inlays) */}
        {[3, 5, 7, 9, 12, 15].map((markerFret) => (
          <Box
            key={`marker-${markerFret}`}
            sx={{
              position: 'absolute',
              bottom: 5,
              left: fretSpacing * markerFret - (fretSpacing / 2),
              width: 20,
              height: 20,
              borderRadius: '50%',
              backgroundColor: markerFret === 12 ? 'rgba(103, 58, 183, 0.8)' : 'rgba(103, 58, 183, 0.5)',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.75rem',
              fontWeight: 'bold',
              zIndex: 3
            }}
          >
            {markerFret}
          </Box>
        ))}
        
        {/* Strings */}
        {dimensions.height > 0 && strings.map((string, i) => (
          <Box
            key={`string-${i}`}
            sx={{
              position: 'absolute',
              top: `${(i * stringSpacing) / dimensions.height * 100}%`,
              left: 0,
              width: '100%',
              height: i < 3 ? 1 : (i < 5 ? 2 : 3), // Varying thickness
              backgroundColor: i === 0 ? '#e53935' : // E string (high)
                             i === 1 ? '#fb8c00' : // B string
                             i === 2 ? '#fdd835' : // G string
                             i === 3 ? '#43a047' : // D string
                             i === 4 ? '#1e88e5' : // A string
                                       '#8e24aa',   // E string (low)
              boxShadow: '0 0 2px rgba(0,0,0,0.3)',
              zIndex: 2
            }}
          />
        ))}
        
        {/* String labels */}
        {dimensions.height > 0 && strings.map((string, i) => (
          <Typography
            key={`label-${i}`}
            variant="caption"
            sx={{
              position: 'absolute',
              top: `${(i * stringSpacing) / dimensions.height * 100}%`,
              left: '-20px',
              transform: 'translateY(-50%)',
              fontSize: '12px',
              fontWeight: 'bold',
              color: i === 0 ? '#e53935' : // E string (high)
                     i === 1 ? '#fb8c00' : // B string
                     i === 2 ? '#fdd835' : // G string
                     i === 3 ? '#43a047' : // D string
                     i === 4 ? '#1e88e5' : // A string
                               '#8e24aa',   // E string (low)
              zIndex: 3
            }}
          >
            {string}
          </Typography>
        ))}
        
        {/* Active notes with enhanced styling and animation */}
        {dimensions.width > 0 && dimensions.height > 0 && activeNotes.map((note, index) => {
          const position = getNotePosition(note.note);
          if (!position) return null;
          
          // Calculate time position for animation timing
          const notePlaying = currentTime >= note.time && currentTime <= (note.time + note.duration);
          const noteAboutToPlay = currentTime < note.time && currentTime >= (note.time - 0.2);
          const noteFadingOut = currentTime > (note.time + note.duration) && currentTime <= (note.time + note.duration + 0.3);
          
          // Calculate opacity based on note timing
          let opacity = 1;
          if (noteAboutToPlay) {
            // Fade in as note approaches
            opacity = 0.4 + 0.6 * (1 - (note.time - currentTime) / 0.2);
          } else if (noteFadingOut) {
            // Fade out after note ends
            opacity = 0.8 * (1 - (currentTime - (note.time + note.duration)) / 0.3);
          }
          
          // Get note color based on note name (C, D, E, etc.)
          const noteName = note.note.replace(/\d+$/, '');
          const noteColors = {
            'C': '#e53935', // Red
            'C#': '#d81b60', 'Db': '#d81b60', // Pink
            'D': '#8e24aa', // Purple
            'D#': '#5e35b1', 'Eb': '#5e35b1', // Deep Purple
            'E': '#3949ab', // Indigo
            'F': '#1e88e5', // Blue
            'F#': '#039be5', 'Gb': '#039be5', // Light Blue
            'G': '#00acc1', // Cyan
            'G#': '#00897b', 'Ab': '#00897b', // Teal
            'A': '#43a047', // Green
            'A#': '#7cb342', 'Bb': '#7cb342', // Light Green
            'B': '#c0ca33', // Lime
          };
          const noteColor = noteColors[noteName] || '#673ab7';
          
          const noteSize = notePlaying ? 20 : 16; // Bigger when actually playing
          
          // Add chord information if available
          const chordDisplay = note.chord ? (
            <Chip
              size="small"
              label={note.chord}
              sx={{
                position: 'absolute',
                top: -25,
                left: -20,
                fontSize: '0.65rem',
                height: 18,
                backgroundColor: 'rgba(255,255,255,0.85)',
                border: `1px solid ${noteColor}`,
                color: noteColor,
                fontWeight: 'bold',
                zIndex: 4
              }}
            />
          ) : null;
          
          return (
            <Box
              key={`note-${index}`}
              sx={{
                position: 'absolute',
                top: `${(position.string * stringSpacing) / dimensions.height * 100}%`,
                left: `${((position.fret - 0.5) * fretSpacing) / dimensions.width * 100}%`,
                width: noteSize,
                height: noteSize,
                borderRadius: '50%',
                backgroundColor: noteColor,
                opacity: opacity,
                zIndex: 3,
                boxShadow: notePlaying ? `0 0 10px ${noteColor}` : '0 0 5px rgba(0,0,0,0.3)',
                transition: 'all 0.15s ease-in-out',
                transform: notePlaying ? 'scale(1.1)' : 'scale(1)'
              }}
            >
              {chordDisplay}
              <Typography
                variant="caption"
                sx={{
                  position: 'absolute',
                  bottom: -18,
                  left: 0,
                  width: noteSize,
                  textAlign: 'center',
                  color: noteColor,
                  fontWeight: 'bold',
                  fontSize: '0.7rem',
                  textShadow: '0px 0px 3px white',
                }}
              >
                {noteName}
              </Typography>
            </Box>
          );
        })}
        
        {/* Enhanced floating instruction box */}
        <Fade in={true}>
          <Box sx={{
            position: 'absolute',
            bottom: -50,
            right: 20,
            padding: '10px 15px',
            borderRadius: '12px',
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            color: 'white',
            maxWidth: '60%',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 8px 25px rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            fontSize: '12px',
            transform: 'translateY(-100%)',
            animation: 'float 4s ease-in-out infinite',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            '@keyframes float': {
              '0%': { transform: 'translateY(-100%)' },
              '50%': { transform: 'translateY(-115%)' },
              '100%': { transform: 'translateY(-100%)' }
            }
          }}>
            <InfoIcon sx={{ fontSize: 16, color: '#2196f3' }} />
            <Box>
              <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block' }}>
                Interactive Fretboard
              </Typography>
              <Typography variant="caption" sx={{ fontSize: '10px' }}>
                Notes are displayed with color-coding by string. Watch the progress ring for note duration.
              </Typography>
            </Box>
          </Box>
        </Fade>
        
        {/* Legend floating box */}
        <Fade in={true}>
          <Box sx={{
            position: 'absolute',
            bottom: -50,
            left: 20,
            padding: '10px 15px',
            borderRadius: '12px',
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            color: 'white',
            maxWidth: '40%',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 8px 25px rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            fontSize: '12px',
            transform: 'translateY(-100%)',
            animation: 'float-alt 4.5s ease-in-out infinite',
            '@keyframes float-alt': {
              '0%': { transform: 'translateY(-100%)' },
              '50%': { transform: 'translateY(-110%)' },
              '100%': { transform: 'translateY(-100%)' }
            }
          }}>
            <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>
              String Colors:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {['High E', 'B', 'G', 'D', 'A', 'Low E'].map((string, i) => (
                <Chip 
                  key={string}
                  label={string} 
                  size="small"
                  sx={{ 
                    height: '18px', 
                    fontSize: '9px',
                    backgroundColor: [
                      'rgba(255, 89, 94, 0.7)',   // High E
                      'rgba(255, 202, 58, 0.7)',  // B
                      'rgba(138, 201, 38, 0.7)',  // G
                      'rgba(25, 130, 196, 0.7)',  // D
                      'rgba(106, 76, 147, 0.7)',  // A
                      'rgba(255, 89, 94, 0.7)',   // Low E
                    ][i],
                    color: 'white'
                  }}
                />
              ))}
            </Box>
          </Box>
        </Fade>
      </Box>
    </Paper>
  );
};

export default GuitarFretboard;
