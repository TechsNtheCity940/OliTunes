import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, Fade, Zoom, Chip, LinearProgress, Card } from '@mui/material';
import MusicNoteIcon from '@mui/icons-material/MusicNote';
import TuneIcon from '@mui/icons-material/Tune';
import InfoIcon from '@mui/icons-material/Info';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';

const AnimatedString = ({ stringName, tabLine, currentBeatPosition, activeFrets }) => {
  // Find which positions have frets played
  const fretPositions = [];
  
  for (let i = 0; i < tabLine.length; i++) {
    const char = tabLine[i];
    if (char !== '-' && char !== '|' && char !== '*') {
      let fretNum = '';
      // Handle multi-digit frets
      while (i < tabLine.length && tabLine[i] !== '-' && tabLine[i] !== '|' && tabLine[i] !== '*') {
        fretNum += tabLine[i];
        i++;
      }
      i--; // Adjust for the extra increment
      
      fretPositions.push({
        pos: i - fretNum.length + 1,
        fret: fretNum.trim()
      });
    }
  }

  // Determine if a position is a beat marker
  const isBeatMarker = (idx) => {
    return tabLine[idx] === '|' || tabLine[idx] === '*';
  };

  // Calculate if this string is actively being played
  const isStringActive = activeFrets.some(fret => 
    fret.string.toLowerCase() === stringName.toLowerCase()
  );

  // String color mapping
  const stringColors = {
    'e': '#FF595E', // High E - Red
    'B': '#FFCA3A', // B - Yellow
    'G': '#8AC926', // G - Green
    'D': '#1982C4', // D - Blue
    'A': '#6A4C93', // A - Purple
    'E': '#FF595E', // Low E - Red again
  };
  
  const stringColor = stringColors[stringName] || '#888';

  return (
    <Box sx={{ 
      position: 'relative', 
      my: 0.7,
      height: '24px',
      display: 'flex',
      alignItems: 'center',
      opacity: isStringActive ? 1 : 0.8,
      transition: 'all 0.3s ease',
      transform: isStringActive ? 'scale(1.02)' : 'scale(1)'
    }}>
      {/* String name with colored circle */}
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        width: '30px',
        mr: 1
      }}>
        <FiberManualRecordIcon 
          sx={{ 
            fontSize: 14, 
            mr: 0.5,
            color: stringColor,
            opacity: isStringActive ? 1 : 0.5,
            filter: isStringActive ? 'drop-shadow(0 0 2px rgba(255,255,255,0.7))' : 'none',
            animation: isStringActive ? 'pulse 1s infinite' : 'none',
            '@keyframes pulse': {
              '0%': { opacity: 0.7 },
              '50%': { opacity: 1 },
              '100%': { opacity: 0.7 }
            }
          }} 
        />
        <Typography 
          variant="body2" 
          sx={{ 
            fontWeight: 'bold',
            color: isStringActive ? stringColor : 'text.secondary',
            transition: 'color 0.3s ease',
            textShadow: isStringActive ? '0 0 8px rgba(0,0,0,0.1)' : 'none'
          }}
        >
          {stringName}
        </Typography>
      </Box>
      
      {/* Tablature line with enhanced styling */}
      <Box sx={{ 
        position: 'relative',
        flexGrow: 1, 
        height: '100%',
        backgroundColor: 'rgba(0,0,0,0.03)',
        borderRadius: '6px',
        overflow: 'hidden',
        fontFamily: 'monospace',
        border: '1px solid',
        borderColor: isStringActive ? `${stringColor}33` : 'rgba(0,0,0,0.08)',
        boxShadow: isStringActive ? `0 0 10px ${stringColor}22` : 'none',
        transition: 'all 0.3s ease'
      }}>
        {/* Current position indicator */}
        <Box sx={{
          position: 'absolute',
          left: `${currentBeatPosition * 100}%`,
          top: 0,
          height: '100%',
          width: '2px',
          backgroundColor: '#f50057',
          boxShadow: '0 0 12px #f50057',
          transform: 'translateX(-50%)',
          zIndex: 10
        }} />
        
        {/* Beat markers as vertical lines */}
        {Array.from({ length: tabLine.length }).map((_, idx) => (
          isBeatMarker(idx) && (
            <Box
              key={`marker-${idx}`}
              sx={{
                position: 'absolute',
                left: `${(idx / tabLine.length) * 100}%`,
                top: 0,
                height: '100%',
                width: '1px',
                backgroundColor: tabLine[idx] === '|' ? 'rgba(0,0,0,0.25)' : 'rgba(0,0,0,0.1)',
                zIndex: 1
              }}
            />
          )
        ))}
        
        {/* Fret numbers with enhanced animations */}
        {fretPositions.map((pos, idx) => {
          const isActive = activeFrets.some(fret => 
            fret.string.toLowerCase() === stringName.toLowerCase() && 
            fret.fret === pos.fret
          );
          
          // Calculate the left position as a percentage
          const leftPos = (pos.pos / tabLine.length) * 100;
          
          // Get note display from active frets
          const activeNote = activeFrets.find(fret => 
            fret.string.toLowerCase() === stringName.toLowerCase() && 
            fret.fret === pos.fret
          );
          
          return (
            <React.Fragment key={`fret-${idx}`}>
              {/* Base fret marker */}
              <Box sx={{
                position: 'absolute',
                left: `${leftPos}%`,
                top: '50%',
                transform: 'translate(-50%, -50%)',
                fontFamily: 'monospace',
                fontWeight: 'bold',
                fontSize: '14px',
                color: isActive ? 'white' : 'text.primary',
                zIndex: 3,
                textShadow: isActive ? '0 0 4px rgba(0,0,0,0.5)' : 'none'
              }}>
                {pos.fret}
              </Box>
              
              {/* Active fret highlight with pulse animation */}
              {isActive && (
                <Zoom in={true} timeout={300}>
                  <Box sx={{
                    position: 'absolute',
                    left: `${leftPos}%`,
                    top: '50%',
                    width: '22px',
                    height: '22px',
                    borderRadius: '50%',
                    backgroundColor: stringColor,
                    transform: 'translate(-50%, -50%)',
                    boxShadow: `0 0 10px ${stringColor}aa`,
                    animation: 'pulse 1.5s infinite',
                    zIndex: 2,
                    '&::before': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      borderRadius: '50%',
                      background: `radial-gradient(circle at 30% 30%, 
                        ${stringColor} 0%, 
                        ${stringColor}88 60%, 
                        transparent 70%
                      )`,
                      opacity: 0.9,
                      zIndex: -1
                    },
                    '@keyframes pulse': {
                      '0%': {
                        boxShadow: `0 0 0 0 ${stringColor}aa`,
                        transform: 'translate(-50%, -50%) scale(1)'
                      },
                      '70%': {
                        boxShadow: `0 0 0 8px ${stringColor}00`,
                        transform: 'translate(-50%, -50%) scale(1.1)'
                      },
                      '100%': {
                        boxShadow: `0 0 0 0 ${stringColor}00`,
                        transform: 'translate(-50%, -50%) scale(1)'
                      }
                    }
                  }}/>
                </Zoom>
              )}
              
              {/* Floating note indicator above active fret */}
              {isActive && activeNote && (
                <Fade in={true}>
                  <Box sx={{
                    position: 'absolute',
                    left: `${leftPos}%`,
                    top: '-100%',
                    transform: 'translate(-50%, -50%)',
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    color: 'white',
                    padding: '2px 5px',
                    borderRadius: '4px',
                    fontSize: '10px',
                    fontWeight: 'bold',
                    zIndex: 5,
                    boxShadow: '0 2px 5px rgba(0,0,0,0.3)',
                    border: `1px solid ${stringColor}88`,
                    animation: 'float 3s ease-in-out infinite',
                    '@keyframes float': {
                      '0%': { transform: 'translate(-50%, -50%)' },
                      '50%': { transform: 'translate(-50%, -60%)' },
                      '100%': { transform: 'translate(-50%, -50%)' }
                    }
                  }}>
                    {activeNote.note}
                  </Box>
                </Fade>
              )}
            </React.Fragment>
          );
        })}
      </Box>
    </Box>
  );
};

const TablatureDisplay = ({ tablature, currentTime, notePositions }) => {
  const [currentBar, setCurrentBar] = useState(0);
  const [currentMeasure, setCurrentMeasure] = useState(0);
  const [activeFrets, setActiveFrets] = useState([]);
  const containerRef = useRef(null);
  
  // Find the current bar and measure based on currentTime
  useEffect(() => {
    if (!tablature || !tablature.bars || tablature.bars.length === 0) return;
    
    // Find current bar
    const barIndex = tablature.bars.findIndex(bar => 
      currentTime >= bar.startTime && currentTime < bar.endTime
    );
    
    if (barIndex !== -1) {
      setCurrentBar(barIndex);
      
      // Find current measure in the bar
      const bar = tablature.bars[barIndex];
      const measureIndex = bar.measures.findIndex(measure => 
        currentTime >= measure.startTime && currentTime < measure.endTime
      );
      
      if (measureIndex !== -1) {
        setCurrentMeasure(measureIndex);
      }
    }
    
    // Find active notes/frets
    if (notePositions) {
      const active = notePositions.filter(note => 
        currentTime >= note.time && currentTime <= (note.time + note.duration)
      ).map(note => ({
        string: ['e', 'B', 'G', 'D', 'A', 'E'][note.string],
        fret: note.fret.toString(),
        note: note.note
      }));
      
      setActiveFrets(active);
    }
  }, [currentTime, tablature, notePositions]);
  
  // Scroll to keep the current measure in view
  useEffect(() => {
    if (containerRef.current && currentBar >= 0 && currentMeasure >= 0) {
      const measureElement = document.getElementById(`measure-${currentBar}-${currentMeasure}`);
      if (measureElement) {
        containerRef.current.scrollTo({
          top: measureElement.offsetTop - 100,
          behavior: 'smooth'
        });
      }
    }
  }, [currentBar, currentMeasure]);
  
  if (!tablature || !tablature.bars || tablature.bars.length === 0) {
    return (
      <Paper 
        elevation={3} 
        sx={{ 
          p: 3, 
          my: 2,
          background: 'linear-gradient(145deg, #ffffff, #f0f0f0)',
          borderRadius: '12px'
        }}
      >
        <Box sx={{ textAlign: 'center' }}>
          <TuneIcon sx={{ fontSize: 40, color: '#bbb', mb: 2 }} />
          <Typography variant="h6" sx={{ color: '#666' }}>Tablature not available</Typography>
          <Typography variant="body2" sx={{ color: '#888', mt: 1 }}>
            Click "Analyze and Generate Tablature" to start
          </Typography>
        </Box>
      </Paper>
    );
  }
  
  // Calculate current position within measure for playhead
  const getCurrentBeatPosition = () => {
    if (currentBar >= 0 && currentMeasure >= 0 && tablature.bars[currentBar]?.measures[currentMeasure]) {
      const measure = tablature.bars[currentBar].measures[currentMeasure];
      const duration = measure.endTime - measure.startTime;
      return (currentTime - measure.startTime) / duration;
    }
    return 0;
  };
  
  // Calculate overall playback progress
  const getOverallProgress = () => {
    if (!tablature || !tablature.totalDuration) return 0;
    return (currentTime / tablature.totalDuration) * 100;
  };
  
  const renderMeasuresInRows = (bars) => {
    const allMeasures = [];
    
    // Flatten all measures with their bar information
    bars.forEach((bar, barIndex) => {
      bar.measures.forEach((measure, measureIndex) => {
        allMeasures.push({
          bar: barIndex,
          measure: measureIndex,
          data: measure,
          barData: bar
        });
      });
    });
    
    // Calculate how many rows we need
    const totalRows = Math.ceil(allMeasures.length / 4);
    
    return Array.from({ length: totalRows }).map((_, rowIndex) => {
      const startIndex = rowIndex * 4;
      const rowMeasures = allMeasures.slice(startIndex, startIndex + 4);
      
      return (
        <Box 
          key={`row-${rowIndex}`}
          sx={{ 
            display: 'flex', 
            width: '100%', 
            mb: 3,
            justifyContent: 'flex-start',
            gap: 1
          }}
        >
          {rowMeasures.map((item) => (
            <Box 
              key={`measure-${item.bar}-${item.measure}`}
              ref={el => {
                if (item.bar === currentBar && item.measure === currentMeasure) {
                  currentMeasureRef.current = el;
                }
              }}
              sx={{
                width: `calc(${100 / 4}% - ${(4 - 1) * 8 / 4}px)`,
                minHeight: 130,
                position: 'relative',
                border: '1px solid #ddd',
                borderRadius: '4px',
                p: 1,
                backgroundColor: (item.bar === currentBar && item.measure === currentMeasure) 
                  ? 'rgba(103, 58, 183, 0.05)' 
                  : '#fff',
                boxShadow: (item.bar === currentBar && item.measure === currentMeasure)
                  ? '0 0 5px rgba(103, 58, 183, 0.3)'
                  : 'none',
                '&:hover': {
                  boxShadow: '0 2px 5px rgba(0,0,0,0.1)'
                },
                display: 'flex',
                flexDirection: 'column'
              }}
            >
              {/* Bar and Chord Information */}
              <Box sx={{ 
                mb: 1, 
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <Typography 
                  variant="caption" 
                  sx={{ 
                    fontWeight: 'bold',
                    color: '#666'
                  }}
                >
                  Bar {item.bar + 1}.{item.measure + 1}
                </Typography>
                
                {item.barData.chord && (
                  <Chip 
                    label={item.barData.chord} 
                    size="small" 
                    color="primary" 
                    sx={{ fontWeight: 'bold', height: 22, fontSize: '0.7rem' }} 
                  />
                )}
              </Box>
              
              {/* Guitar strings */}
              <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                {['E', 'B', 'G', 'D', 'A', 'E'].map((string, stringIndex) => (
                  <Box 
                    key={`string-${item.bar}-${item.measure}-${stringIndex}`}
                    sx={{ 
                      position: 'relative', 
                      height: '18px',
                      display: 'flex',
                      alignItems: 'center',
                      borderBottom: '1px solid #ddd'
                    }}
                  >
                    {/* String label */}
                    <Typography variant="caption" sx={{ 
                      position: 'absolute', 
                      left: -18, 
                      fontSize: '10px', 
                      color: '#666',
                      fontWeight: 'bold'
                    }}>
                      {string}
                    </Typography>
                    
                    {/* Notes on this string */}
                    {item.data.notes.filter(note => note.string === stringIndex).map((note, noteIndex) => (
                      <Box
                        key={`note-${item.bar}-${item.measure}-${stringIndex}-${noteIndex}`}
                        sx={{
                          position: 'absolute',
                          left: `${note.position * 100}%`,
                          transform: 'translateX(-50%)',
                          fontSize: '14px',
                          fontWeight: 'bold',
                          color: '#333',
                          backgroundColor: note.isHighlighted ? 'rgba(103, 58, 183, 0.2)' : 'transparent',
                          borderRadius: '50%',
                          width: '20px',
                          height: '20px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          zIndex: 5
                        }}
                      >
                        {note.fret}
                      </Box>
                    ))}
                  </Box>
                ))}
              </Box>
              
              {/* Playback indicator */}
              {(item.bar === currentBar && item.measure === currentMeasure) && (
                <Box
                  sx={{
                    position: 'absolute',
                    height: '100%',
                    width: '2px',
                    backgroundColor: 'rgba(103, 58, 183, 0.7)',
                    top: 0,
                    left: `${getCurrentBeatPosition() * 100}%`,
                    transform: 'translateX(-50%)',
                    zIndex: 10,
                    boxShadow: '0 0 5px rgba(103, 58, 183, 0.5)'
                  }}
                />
              )}
            </Box>
          ))}
        </Box>
      );
    });
  };

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        my: 2, 
        overflow: 'auto',
        position: 'relative',
        maxHeight: 800,
        width: '100%',
        backgroundColor: '#f8f8f8',
        boxShadow: '0 3px 10px rgba(0, 0, 0, 0.1)',
        mx: 'auto'
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <MusicNoteIcon sx={{ mr: 1, color: '#1976d2' }} />
        <Typography variant="h6" sx={{ fontWeight: 500 }}>
          Interactive Tablature
        </Typography>
      </Box>
      
      {/* Overall progress bar */}
      <LinearProgress 
        variant="determinate" 
        value={getOverallProgress()} 
        sx={{ 
          mb: 3, 
          height: 6, 
          borderRadius: 3,
          backgroundColor: 'rgba(0,0,0,0.05)',
          '& .MuiLinearProgress-bar': {
            backgroundImage: 'linear-gradient(to right, #1976d2, #f50057)'
          }
        }}
      />
      
      {/* Time display floating box */}
      <Fade in={true}>
        <Box sx={{
          position: 'absolute',
          top: 15,
          right: 15,
          padding: '6px 12px',
          borderRadius: '20px',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          backdropFilter: 'blur(8px)',
          zIndex: 10,
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          animation: 'float 4s ease-in-out infinite',
          display: 'flex',
          alignItems: 'center',
          '@keyframes float': {
            '0%': { transform: 'translateY(0px)' },
            '50%': { transform: 'translateY(-5px)' },
            '100%': { transform: 'translateY(0px)' }
          }
        }}>
          <FiberManualRecordIcon 
            sx={{ 
              fontSize: 12, 
              mr: 1, 
              color: '#f50057',
              animation: 'pulse-me 1s infinite',
              '@keyframes pulse-me': {
                '0%': { opacity: 0.7 },
                '50%': { opacity: 1 },
                '100%': { opacity: 0.7 }
              }
            }} 
          />
          <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
            {Math.floor(currentTime / 60)}:{Math.floor(currentTime % 60).toString().padStart(2, '0')}
          </Typography>
        </Box>
      </Fade>
      
      {/* Active notes floating counter */}
      {activeFrets.length > 0 && (
        <Fade in={true}>
          <Box sx={{
            position: 'absolute',
            top: 15,
            left: 15,
            padding: '6px 12px',
            borderRadius: '20px',
            backgroundColor: 'rgba(25, 118, 210, 0.8)',
            color: 'white',
            backdropFilter: 'blur(8px)',
            zIndex: 10,
            boxShadow: '0 4px 15px rgba(25, 118, 210, 0.3)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            display: 'flex',
            alignItems: 'center'
          }}>
            <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
              Notes: {activeFrets.length}
            </Typography>
          </Box>
        </Fade>
      )}
      
      <Box 
        ref={containerRef}
        sx={{ 
          maxHeight: 800, 
          overflowY: 'auto',
          mt: 2,
          pb: 2,
          '&::-webkit-scrollbar': {
            width: '10px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'rgba(103, 58, 183, 0.05)',
            borderRadius: '10px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: 'rgba(103, 58, 183, 0.3)',
            borderRadius: '10px',
            '&:hover': {
              background: 'rgba(103, 58, 183, 0.5)'
            }
          }
        }}
      >
        {renderMeasuresInRows(tablature.bars)}
      </Box>
      
      {/* Enhanced help box with icon */}
      <Fade in={true}>
        <Box sx={{
          position: 'absolute',
          bottom: 15,
          left: 15,
          padding: '10px 15px',
          borderRadius: '12px',
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          color: 'white',
          backdropFilter: 'blur(10px)',
          zIndex: 5,
          maxWidth: '250px',
          boxShadow: '0 6px 20px rgba(0, 0, 0, 0.4)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          animation: 'float-alt 4.5s ease-in-out infinite',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '8px',
          '@keyframes float-alt': {
            '0%': { transform: 'translateY(0px)' },
            '50%': { transform: 'translateY(5px)' },
            '100%': { transform: 'translateY(0px)' }
          }
        }}>
          <InfoIcon sx={{ fontSize: 16, color: '#2196f3', mt: 0.3 }} />
          <Box>
            <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5 }}>
              How to read tablature:
            </Typography>
            <Typography variant="caption" component="div" sx={{ fontSize: '10px', opacity: 0.9, lineHeight: 1.4 }}>
              Numbers show which fret to play on each string. The strings are labeled with 
              letters (e-E). The red line tracks current playback position. Glowing numbers
              indicate notes currently playing.
            </Typography>
          </Box>
        </Box>
      </Fade>
    </Paper>
  );
};

export default TablatureDisplay;
