import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  IconButton, 
  Slider,
  Paper,
  Tooltip,
  useTheme,
  Fade,
  Zoom,
  Grid,
  Divider,
  Button
} from '@mui/material';
import { 
  Add, 
  Remove, 
  MusicNote, 
  GraphicEq,
  Brush,
  FormatColorFill,
  FiberManualRecord
} from '@mui/icons-material';

const BeatPatternEditor = ({ 
  pattern, 
  onPatternChange, 
  subdivision,
  onSubdivisionChange,
  accents,
  onAccentChange 
}) => {
  const theme = useTheme();
  const [isEditingColors, setIsEditingColors] = useState(false);
  const [beatColors, setBeatColors] = useState(
    pattern.map(() => theme.palette.primary.main)
  );

  // Predefined accent color palette
  const accentColors = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    '#FF5722', // Deep Orange
    '#8BC34A', // Light Green
    '#FFEB3B', // Yellow
    '#E91E63', // Pink
  ];

  const handleAccentChange = (beatIndex, subdivisionIndex, value) => {
    const newAccents = [...accents];
    newAccents[beatIndex][subdivisionIndex] = value;
    onAccentChange(newAccents);
  };

  const handleColorChange = (beatIndex, color) => {
    const newColors = [...beatColors];
    newColors[beatIndex] = color;
    setBeatColors(newColors);
  };

  const getAccentColor = (value) => {
    if (value >= 80) return theme.palette.error.main;
    if (value >= 50) return theme.palette.warning.main;
    if (value >= 30) return theme.palette.info.main;
    return theme.palette.text.secondary;
  };

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 3, 
        mt: 2, 
        borderRadius: 2,
        background: 'linear-gradient(145deg, rgba(30,30,30,0.9) 0%, rgba(40,40,40,0.8) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255,255,255,0.05)',
        overflow: 'hidden',
        position: 'relative',
      }}
      className="music-paper"
    >
      {/* Background music notes pattern */}
      <Box sx={{ 
        position: 'absolute', 
        top: 0, 
        left: 0, 
        right: 0, 
        bottom: 0, 
        opacity: 0.05, 
        backgroundImage: 'url(/music-pattern.svg)',
        backgroundSize: '200px',
        zIndex: 0,
        pointerEvents: 'none'
      }} />

      <Box sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          mb: 3
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <GraphicEq sx={{ 
              color: theme.palette.primary.main, 
              mr: 1.5, 
              fontSize: 28 
            }} />
            <Typography 
              variant="h5" 
              className="shimmer" 
              fontWeight="bold"
            >
              Beat Pattern Editor
            </Typography>
          </Box>
          
          <Box>
            <Tooltip title={isEditingColors ? "Exit color mode" : "Edit beat colors"}>
              <IconButton 
                onClick={() => setIsEditingColors(!isEditingColors)}
                color={isEditingColors ? "secondary" : "default"}
                size="small"
                sx={{ mr: 1 }}
              >
                <Brush />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        <Divider sx={{ mb: 3, opacity: 0.3 }} />
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Paper
              elevation={1}
              sx={{
                p: 2,
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: theme.shape.borderRadius,
              }}
            >
              <Typography variant="subtitle2" gutterBottom fontWeight="medium">
                Beat Settings
              </Typography>
              
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 1.5, 
                mb: 3,
                justifyContent: 'center',
                mt: 2
              }}>
                <Typography variant="body2">Subdivisions:</Typography>
                <IconButton 
                  size="small" 
                  onClick={() => onSubdivisionChange(Math.max(1, subdivision - 1))}
                  sx={{ 
                    backgroundColor: 'rgba(0,0,0,0.2)', 
                    '&:hover': { backgroundColor: 'rgba(0,0,0,0.3)' } 
                  }}
                >
                  <Remove fontSize="small" />
                </IconButton>
                
                <Box sx={{ 
                  minWidth: '36px', 
                  height: '36px', 
                  borderRadius: '4px', 
                  backgroundColor: 'rgba(0,0,0,0.3)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold'
                }}>
                  <Typography variant="body1">{subdivision}</Typography>
                </Box>
                
                <IconButton 
                  size="small" 
                  onClick={() => onSubdivisionChange(Math.min(4, subdivision + 1))}
                  sx={{ 
                    backgroundColor: 'rgba(0,0,0,0.2)', 
                    '&:hover': { backgroundColor: 'rgba(0,0,0,0.3)' } 
                  }}
                >
                  <Add fontSize="small" />
                </IconButton>
              </Box>
              
              <Box sx={{ mt: 4 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Beat Visualization
                </Typography>
                <Box sx={{ 
                  display: 'flex', 
                  flexDirection: 'row', 
                  gap: 1, 
                  mt: 1,
                  justifyContent: 'center',
                  flexWrap: 'wrap'
                }}>
                  {pattern.map((_, idx) => (
                    <Box 
                      key={`viz-${idx}`}
                      sx={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '50%',
                        backgroundColor: beatColors[idx],
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        animation: `beat 1s ${idx * 0.2}s infinite`,
                        transform: `scale(${1 + accents[idx].reduce((sum, val) => sum + val, 0) / (subdivision * 200)})`,
                        transition: 'transform 0.3s ease, background-color 0.3s ease',
                        boxShadow: '0 4px 8px rgba(0,0,0,0.3)'
                      }}
                    >
                      <Typography fontWeight="bold" color="white">
                        {idx + 1}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={8}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {pattern.map((beat, beatIndex) => (
                <Zoom 
                  key={beatIndex} 
                  in={true} 
                  style={{ transitionDelay: `${beatIndex * 100}ms` }}
                >
                  <Paper 
                    sx={{ 
                      p: 2,
                      backgroundColor: 'rgba(0,0,0,0.2)',
                      borderRadius: theme.shape.borderRadius,
                      borderLeft: '4px solid',
                      borderLeftColor: beatColors[beatIndex],
                      position: 'relative',
                      overflow: 'hidden',
                      transition: 'all 0.3s ease'
                    }}
                  >
                    {/* Color picker UI */}
                    <Fade in={isEditingColors}>
                      <Box 
                        sx={{ 
                          position: 'absolute',
                          top: 0,
                          right: 0,
                          bottom: 0,
                          width: '100px',
                          background: 'rgba(0,0,0,0.5)',
                          display: isEditingColors ? 'flex' : 'none',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 1,
                          p: 1,
                          zIndex: 5
                        }}
                      >
                        {accentColors.map(color => (
                          <Tooltip key={color} title="Apply color">
                            <IconButton 
                              size="small"
                              onClick={() => handleColorChange(beatIndex, color)}
                              sx={{ 
                                backgroundColor: color,
                                '&:hover': { 
                                  backgroundColor: color,
                                  transform: 'scale(1.1)'
                                },
                                width: '28px',
                                height: '28px'
                              }}
                            >
                              {beatColors[beatIndex] === color && (
                                <FiberManualRecord sx={{ fontSize: 10, color: 'white' }} />
                              )}
                            </IconButton>
                          </Tooltip>
                        ))}
                      </Box>
                    </Fade>
                    
                    <Box sx={{ 
                      display: 'flex', 
                      alignItems: 'center',
                      gap: 2,
                      mr: isEditingColors ? 12 : 0,
                      transition: 'margin-right 0.3s ease'
                    }}>
                      <Box sx={{ 
                        width: '40px',
                        height: '40px',
                        borderRadius: '50%',
                        backgroundColor: beatColors[beatIndex],
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
                      }}>
                        <Typography fontWeight="bold" color="white">
                          {beatIndex + 1}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ 
                        display: 'flex', 
                        gap: 1, 
                        flex: 1,
                        alignItems: 'flex-end'
                      }}>
                        {Array(subdivision).fill(0).map((_, subdivIndex) => {
                          const accentValue = accents[beatIndex][subdivIndex];
                          return (
                            <Box 
                              key={subdivIndex}
                              sx={{ 
                                flex: 1,
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                position: 'relative'
                              }}
                            >
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  position: 'absolute',
                                  top: -20,
                                  color: getAccentColor(accentValue),
                                  fontWeight: 'bold'
                                }}
                              >
                                {accentValue}
                              </Typography>
                              
                              <Slider
                                orientation="vertical"
                                value={accentValue}
                                onChange={(_, value) => handleAccentChange(beatIndex, subdivIndex, value)}
                                min={0}
                                max={100}
                                sx={{ 
                                  height: 100,
                                  color: getAccentColor(accentValue),
                                  '& .MuiSlider-thumb': {
                                    width: 16,
                                    height: 16,
                                    backgroundColor: getAccentColor(accentValue),
                                    boxShadow: `0 0 8px ${getAccentColor(accentValue)}`,
                                    transition: 'all 0.2s ease',
                                    '&:hover, &.Mui-focusVisible': {
                                      boxShadow: `0 0 12px ${getAccentColor(accentValue)}`,
                                    }
                                  },
                                  '& .MuiSlider-track': {
                                    border: 'none',
                                    boxShadow: `0 0 4px ${getAccentColor(accentValue)}`,
                                  }
                                }}
                              />
                              
                              <Box sx={{ 
                                width: '24px',
                                height: '24px',
                                borderRadius: '50%',
                                backgroundColor: 'rgba(0,0,0,0.3)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                mt: 1
                              }}>
                                <Typography 
                                  variant="caption"
                                  sx={{
                                    fontWeight: accentValue > 50 ? 'bold' : 'normal',
                                    color: accentValue > 50 ? getAccentColor(accentValue) : 'inherit'
                                  }}
                                >
                                  {subdivIndex + 1}
                                </Typography>
                              </Box>
                            </Box>
                          );
                        })}
                      </Box>
                    </Box>
                  </Paper>
                </Zoom>
              ))}
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default BeatPatternEditor;
