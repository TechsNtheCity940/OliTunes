import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Box, Typography, Paper, Zoom, Fade, Chip } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import MusicNoteIcon from '@mui/icons-material/MusicNote';

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
          setDimensions({
            width: entry.contentRect.width,
            height: Math.min(250, entry.contentRect.width / 3.5)
          });
        }
      });
      
      resizeObserver.observe(fretboardRef.current);
      return () => resizeObserver.disconnect();
    }
  }, []);
  
  // Update active notes based on current playback time with look-ahead
  useEffect(() => {
    if (!notes) return;
    
    const lookAheadTime = 0.15; // Look ahead 150ms to prepare animations
    
    const active = notes.filter(note => {
      // Include notes that are currently playing or about to play
      return (currentTime >= note.time - lookAheadTime && 
              currentTime <= (note.time + note.duration));
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
        height: Math.min(250, width / 3.5)
      });
    }
  }, [dimensions.width]);
  
  // Calculate spacing for the fretboard
  const stringSpacing = dimensions.height / (strings.length - 1);
  const fretSpacing = dimensions.width / (numberOfFrets + 1);
  
  return (
    <Paper 
      elevation={3}
      sx={{ 
        p: 2, 
        mb: 3,
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s ease-in-out'
      }}
    >
      <Typography variant="h6" gutterBottom>
        Guitar Fretboard Visualization
      </Typography>
      
      <Box 
        ref={fretboardRef}
        sx={{
          position: 'relative',
          width: '100%',
          height: dimensions.height || 250,
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
        {Array.from({ length: numberOfFrets + 1 }).map((_, i) => (
          <Box
            key={`fret-${i}`}
            sx={{
              position: 'absolute',
              top: 0,
              left: `${(i * fretSpacing) / Math.max(1, dimensions.width) * 100}%`,
              width: i === 0 ? '6px' : '2px',
              height: '100%',
              backgroundColor: i === 0 ? '#444' : '#888', // Nut is darker and wider
              zIndex: 1
            }}
          />
        ))}
        
        {/* Fret markers (inlays) */}
        {dimensions.width > 0 && [3, 5, 7, 9, 12, 15].map(fret => (
          <Box
            key={`marker-${fret}`}
            sx={{
              position: 'absolute',
              top: '50%',
              left: `${((fret - 0.5) * fretSpacing) / dimensions.width * 100}%`,
              width: fret === 12 ? '14px' : '10px',
              height: fret === 12 ? '14px' : '10px',
              borderRadius: '50%',
              backgroundColor: 'rgba(255, 255, 255, 0.7)',
              transform: 'translate(-50%, -50%)',
              zIndex: 1,
              ...(fret === 12 && {
                boxShadow: '0 0 5px rgba(255,255,255,0.5)',
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  top: '200%',
                  left: '50%',
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: 'rgba(255, 255, 255, 0.7)',
                  transform: 'translate(-50%, -50%)'
                }
              })
            }}
          />
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
              height: `${[1, 1, 2, 3, 4, 5][i]}px`, // Varying string thickness
              backgroundColor: '#DDD',
              boxShadow: '0 0 2px rgba(0,0,0,0.3)',
              zIndex: 2
            }}
          />
        ))}
        
        {/* String labels */}
        {dimensions.height > 0 && strings.map((string, i) => (
          <Typography
            key={`label-${i}`}
            sx={{
              position: 'absolute',
              top: `${(i * stringSpacing) / dimensions.height * 100}%`,
              left: '-20px',
              transform: 'translateY(-50%)',
              fontSize: '12px',
              fontWeight: 'bold',
              color: '#666',
              zIndex: 3
            }}
          >
            {string}
          </Typography>
        ))}
        
        {/* Active notes with enhanced animations */}
        {dimensions.width > 0 && dimensions.height > 0 && activeNotes.map((note, index) => {
          const position = getNotePosition(note.note);
          if (!position) return null;
          
          // Calculate if note is about to play or currently playing
          const isPreview = currentTime < note.time;
          const noteProgress = Math.min(1, Math.max(0, 
            (currentTime - note.time) / Math.max(0.1, note.duration)
          ));
          
          // Calculate color based on string number
          const stringColors = [
            'rgba(255, 89, 94, 1)',  // High E - Red
            'rgba(255, 202, 58, 1)', // B - Yellow
            'rgba(138, 201, 38, 1)', // G - Green
            'rgba(25, 130, 196, 1)', // D - Blue
            'rgba(106, 76, 147, 1)', // A - Purple
            'rgba(255, 89, 94, 1)',  // Low E - Red again
          ];
          
          const baseColor = stringColors[position.string];
          
          return (
            <React.Fragment key={`note-${index}`}>
              {/* Main note marker */}
              <Zoom in={true} style={{ transitionDelay: isPreview ? '120ms' : '0ms' }}>
                <Box
                  sx={{
                    position: 'absolute',
                    top: `${(position.string * stringSpacing) / dimensions.height * 100}%`,
                    left: `${((position.fret - 0.5) * fretSpacing) / dimensions.width * 100}%`,
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    backgroundColor: isPreview ? 
                      `rgba(${baseColor.slice(5, -1)}, 0.5)` : 
                      baseColor,
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    transform: 'translate(-50%, -50%) scale(1.2)',
                    zIndex: 10,
                    boxShadow: `0 0 20px ${baseColor.replace('1)', '0.8)')}`,
                    animation: isPreview ? 'preview 1s infinite' : 'pulse 1s infinite',
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      borderRadius: '50%',
                      background: `radial-gradient(circle at 30% 30%, 
                        ${baseColor.replace('1)', '1)')} 0%,
                        ${baseColor.replace('1)', '0.5)')} 60%,
                        transparent 70%
                      )`,
                      opacity: 0.9,
                      zIndex: -1
                    },
                    '@keyframes pulse': {
                      '0%': {
                        boxShadow: `0 0 0 0 ${baseColor.replace('1)', '0.7)')}`,
                        transform: 'translate(-50%, -50%) scale(1.2)'
                      },
                      '70%': {
                        boxShadow: `0 0 0 15px ${baseColor.replace('1)', '0)')}`,
                        transform: 'translate(-50%, -50%) scale(1.45)'
                      },
                      '100%': {
                        boxShadow: `0 0 0 0 ${baseColor.replace('1)', '0)')}`,
                        transform: 'translate(-50%, -50%) scale(1.2)'
                      }
                    },
                    '@keyframes preview': {
                      '0%': {
                        opacity: 0.7,
                        transform: 'translate(-50%, -50%) scale(1.1)'
                      },
                      '50%': {
                        opacity: 0.9,
                        transform: 'translate(-50%, -50%) scale(1.3)'
                      },
                      '100%': {
                        opacity: 0.7,
                        transform: 'translate(-50%, -50%) scale(1.1)'
                      }
                    }
                  }}
                >
                  {position.fret}
                </Box>
              </Zoom>
              
              {/* Enhanced floating note information */}
              <Fade in={!isPreview && note.confidence > 0.6}>
                <Box
                  sx={{
                    position: 'absolute',
                    top: `${(position.string * stringSpacing) / dimensions.height * 100 - 15}%`,
                    left: `${((position.fret - 0.5) * fretSpacing) / dimensions.width * 100 + 8}%`,
                    padding: '6px 10px',
                    borderRadius: '8px',
                    backgroundColor: 'rgba(0, 0, 0, 0.85)',
                    color: 'white',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    zIndex: 11,
                    transform: 'translateY(-100%)',
                    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.4)',
                    backdropFilter: 'blur(8px)',
                    border: `1px solid ${baseColor.replace('1)', '0.6)')}`,
                    animation: 'float 3s ease-in-out infinite',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    '@keyframes float': {
                      '0%': { transform: 'translateY(-100%)' },
                      '50%': { transform: 'translateY(-115%)' },
                      '100%': { transform: 'translateY(-100%)' }
                    }
                  }}
                >
                  <MusicNoteIcon sx={{ fontSize: 14 }} />
                  {note.note}
                  <Chip 
                    label={Math.round(note.confidence * 100) + '%'} 
                    size="small" 
                    sx={{
                      height: '16px',
                      fontSize: '9px',
                      ml: 0.5,
                      backgroundColor: `${baseColor.replace('1)', '0.3)')}`,
                      color: 'white'
                    }}
                  />
                </Box>
              </Fade>
              
              {/* Enhanced progress indicator */}
              {!isPreview && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: `${(position.string * stringSpacing) / dimensions.height * 100}%`,
                    left: `${((position.fret - 0.5) * fretSpacing) / dimensions.width * 100}%`,
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    border: `2px solid ${baseColor}`,
                    opacity: 0.8,
                    transform: 'translate(-50%, -50%)',
                    zIndex: 9,
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      borderRadius: '50%',
                      background: `conic-gradient(
                        ${baseColor} ${noteProgress * 360}deg,
                        transparent ${noteProgress * 360}deg
                      )`,
                      opacity: 0.5
                    },
                    animation: 'ripple 2s linear infinite',
                    '@keyframes ripple': {
                      '0%': {
                        boxShadow: `0 0 0 0px ${baseColor.replace('1)', '0.3)')}`
                      },
                      '100%': {
                        boxShadow: `0 0 0 20px ${baseColor.replace('1)', '0)')}`
                      }
                    }
                  }}
                />
              )}
            </React.Fragment>
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
