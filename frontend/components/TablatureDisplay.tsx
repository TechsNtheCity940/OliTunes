import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, Chip, IconButton, Slider, useTheme } from '@mui/material';
import { ZoomIn, ZoomOut } from '@mui/icons-material';

interface TablatureDisplayProps {
  tablature: any;
  currentTime: number;
  notePositions: any[];
  darkMode?: boolean;
  initialZoom?: number;
}

const TablatureDisplay: React.FC<TablatureDisplayProps> = ({
  tablature,
  currentTime,
  notePositions,
  darkMode = true,
  initialZoom = 1.0
}) => {
  const [zoom, setZoom] = useState(initialZoom);
  const [currentBar, setCurrentBar] = useState<number | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  
  // Colors
  const backgroundColor = darkMode ? '#1a1a1a' : '#f9f9f9';
  const textColor = darkMode ? 'rgba(255, 255, 255, 0.87)' : 'rgba(0, 0, 0, 0.87)';
  const secondaryTextColor = darkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)';
  const borderColor = darkMode ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)';
  const highlightColor = theme.palette.primary.main;
  const barColor = darkMode ? '#333' : '#eaeaea';
  
  // Handle zoom changes
  const handleZoomChange = (event: Event, newValue: number | number[]) => {
    setZoom(newValue as number);
  };
  
  // Update current bar based on playback time
  useEffect(() => {
    if (!tablature || !tablature.bars || tablature.bars.length === 0) return;
    
    const newBar = tablature.bars.findIndex((bar: any) => {
      return currentTime >= bar.startTime && currentTime < bar.endTime;
    });
    
    if (newBar !== -1 && newBar !== currentBar) {
      setCurrentBar(newBar);
      
      // Auto-scroll to the current bar
      if (scrollContainerRef.current && newBar !== null) {
        const barElements = scrollContainerRef.current.querySelectorAll('.tablature-bar');
        if (barElements[newBar]) {
          barElements[newBar].scrollIntoView({
            behavior: 'smooth',
            block: 'nearest',
            inline: 'start'
          });
        }
      }
    }
  }, [currentTime, tablature, currentBar]);
  
  if (!tablature) {
    return (
      <Paper 
        elevation={3} 
        sx={{ 
          p: 3, 
          backgroundColor: darkMode ? '#121212' : '#f7f7f7',
          color: textColor,
          borderRadius: 2
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: 600 }}>Tablature</Typography>
        <Typography variant="body1" sx={{ mt: 2, color: secondaryTextColor }}>
          No tablature available. Analyze an audio file to generate tablature.
        </Typography>
      </Paper>
    );
  }
  
  return (
    <Paper 
      elevation={4} 
      sx={{ 
        p: 2,
        backgroundColor: darkMode ? '#121212' : '#f7f7f7',
        borderRadius: 2
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 600, color: textColor }}>
          Tablature
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}>
            <ZoomOut />
          </IconButton>
          <Slider
            value={zoom}
            min={0.5}
            max={2}
            step={0.1}
            onChange={handleZoomChange}
            aria-labelledby="zoom-slider"
            sx={{ width: '100px', mx: 1 }}
          />
          <IconButton onClick={() => setZoom(Math.min(2, zoom + 0.1))}>
            <ZoomIn />
          </IconButton>
        </Box>
      </Box>
      
      {/* Metadata display */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
        {tablature.tuning && (
          <Chip 
            label={`Tuning: ${tablature.tuning.join(' ')}`.toUpperCase()}
            size="small"
            color="primary"
            sx={{ fontWeight: 500 }}
          />
        )}
        {tablature.timeSignature && (
          <Chip 
            label={`Time: ${tablature.timeSignature}`}
            size="small"
            color="primary"
            variant="outlined"
            sx={{ fontWeight: 500 }}
          />
        )}
        {tablature.tempo && (
          <Chip 
            label={`Tempo: ${Math.round(tablature.tempo)} BPM`}
            size="small"
            color="primary"
            variant="outlined"
            sx={{ fontWeight: 500 }}
          />
        )}
      </Box>
      
      {/* Tablature display */}
      <Box 
        ref={scrollContainerRef}
        sx={{ 
          position: 'relative',
          height: '300px',
          overflowX: 'auto',
          overflowY: 'auto',
          backgroundColor: backgroundColor,
          borderRadius: '8px',
          p: 2,
          '&::-webkit-scrollbar': {
            height: '8px',
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: darkMode ? '#333' : '#ddd',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: darkMode ? '#555' : '#aaa',
            borderRadius: '4px',
            '&:hover': {
              backgroundColor: darkMode ? '#777' : '#999',
            },
          },
        }}
      >
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column',
          fontFamily: 'monospace',
          fontSize: `${14 * zoom}px`,
          whiteSpace: 'nowrap',
        }}>
          {tablature.bars && tablature.bars.map((bar: any, barIdx: number) => (
            <Box 
              key={`bar-${barIdx}`}
              className="tablature-bar"
              sx={{ 
                mb: 3,
                p: 1,
                backgroundColor: currentBar === barIdx ? 
                                  `rgba(${parseInt(highlightColor.slice(1, 3), 16)}, 
                                       ${parseInt(highlightColor.slice(3, 5), 16)}, 
                                       ${parseInt(highlightColor.slice(5, 7), 16)}, 0.15)` : 
                                  'transparent',
                borderRadius: '4px',
                border: `1px solid ${currentBar === barIdx ? highlightColor : 'transparent'}`,
                transition: 'background-color 0.3s ease'
              }}
            >
              <Typography 
                variant="subtitle2" 
                sx={{ 
                  mb: 1, 
                  color: currentBar === barIdx ? highlightColor : secondaryTextColor,
                  fontWeight: currentBar === barIdx ? 'bold' : 'normal'
                }}
              >
                Bar {barIdx + 1} {bar.timeSignature && `(${bar.timeSignature})`}
              </Typography>
              
              {bar.measures && bar.measures.map((measure: any, measureIdx: number) => (
                <Box 
                  key={`measure-${barIdx}-${measureIdx}`}
                  sx={{ 
                    mb: 1,
                    p: 1,
                    backgroundColor: barColor,
                    borderRadius: '4px'
                  }}
                >
                  {/* Simple ASCII tablature display */}
                  {measure.tab && measure.tab.map((line: string, lineIdx: number) => (
                    <Box 
                      key={`tab-line-${barIdx}-${measureIdx}-${lineIdx}`}
                      sx={{ 
                        fontFamily: 'monospace',
                        fontSize: `${16 * zoom}px`,
                        lineHeight: 1.5,
                        whiteSpace: 'pre',
                        color: textColor,
                        letterSpacing: '2px'
                      }}
                    >
                      {line}
                    </Box>
                  ))}
                  
                  {/* If no formatted tablature available, show a placeholder */}
                  {(!measure.tab || measure.tab.length === 0) && (
                    <Box sx={{ py: 1 }}>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: secondaryTextColor,
                          fontStyle: 'italic',
                          textAlign: 'center'
                        }}
                      >
                        Empty measure
                      </Typography>
                    </Box>
                  )}
                </Box>
              ))}
              
              {/* If no measures available, show the raw tablature text */}
              {(!bar.measures || bar.measures.length === 0) && tablature.tablature && (
                <Box 
                  sx={{ 
                    fontFamily: 'monospace',
                    fontSize: `${16 * zoom}px`,
                    lineHeight: 1.5,
                    whiteSpace: 'pre',
                    color: textColor,
                    backgroundColor: barColor,
                    p: 1,
                    borderRadius: '4px'
                  }}
                >
                  {Array.isArray(tablature.tablature) 
                    ? tablature.tablature.join('\n') 
                    : typeof tablature.tablature === 'string' 
                      ? tablature.tablature 
                      : 'No tablature data available'}
                </Box>
              )}
            </Box>
          ))}
          
          {/* Fallback if no bars are available */}
          {(!tablature.bars || tablature.bars.length === 0) && tablature.tablature && (
            <Box 
              sx={{ 
                fontFamily: 'monospace',
                fontSize: `${16 * zoom}px`,
                lineHeight: 1.5,
                whiteSpace: 'pre',
                color: textColor,
                backgroundColor: barColor,
                p: 2,
                borderRadius: '4px'
              }}
            >
              {Array.isArray(tablature.tablature) 
                ? tablature.tablature.join('\n') 
                : typeof tablature.tablature === 'string' 
                  ? tablature.tablature 
                  : 'No tablature data available'}
            </Box>
          )}
        </Box>
      </Box>
      
      {/* Current time indicator */}
      <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="body2" sx={{ color: secondaryTextColor, fontFamily: 'monospace' }}>
          {Math.floor(currentTime / 60)}:{Math.floor(currentTime % 60).toString().padStart(2, '0')}
        </Typography>
        
        <Slider
          value={currentTime}
          min={0}
          max={tablature.totalDuration || 100}
          step={0.1}
          disabled
          sx={{ 
            color: theme.palette.primary.main,
            '& .MuiSlider-thumb': {
              width: 12,
              height: 12,
              transition: '0.3s cubic-bezier(.47,1.64,.41,.8)',
              '&:before': {
                boxShadow: '0 2px 12px 0 rgba(0,0,0,0.4)',
              },
              '&:hover, &.Mui-focusVisible': {
                boxShadow: `0px 0px 0px 8px ${theme.palette.primary.main}33`,
              },
            },
            '& .MuiSlider-rail': {
              opacity: 0.5,
            },
          }}
        />
        
        <Typography variant="body2" sx={{ color: secondaryTextColor, fontFamily: 'monospace' }}>
          {Math.floor((tablature.totalDuration || 0) / 60)}:{Math.floor((tablature.totalDuration || 0) % 60).toString().padStart(2, '0')}
        </Typography>
      </Box>
    </Paper>
  );
};

export default TablatureDisplay;
