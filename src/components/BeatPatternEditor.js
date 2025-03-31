import React from 'react';
import { 
  Box, 
  Typography, 
  IconButton, 
  Slider,
  Paper
} from '@mui/material';
import { Add, Remove } from '@mui/icons-material';

const BeatPatternEditor = ({ 
  pattern, 
  onPatternChange, 
  subdivision,
  onSubdivisionChange,
  accents,
  onAccentChange 
}) => {
  const handleAccentChange = (beatIndex, subdivisionIndex, value) => {
    const newAccents = [...accents];
    newAccents[beatIndex][subdivisionIndex] = value;
    onAccentChange(newAccents);
  };

  return (
    <Paper elevation={1} sx={{ p: 2, mt: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Beat Pattern Editor
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="body2">Subdivisions:</Typography>
        <IconButton 
          size="small" 
          onClick={() => onSubdivisionChange(Math.max(1, subdivision - 1))}
        >
          <Remove />
        </IconButton>
        <Typography variant="body2">{subdivision}</Typography>
        <IconButton 
          size="small" 
          onClick={() => onSubdivisionChange(Math.min(4, subdivision + 1))}
        >
          <Add />
        </IconButton>
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {pattern.map((beat, beatIndex) => (
          <Box 
            key={beatIndex}
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              gap: 2,
              p: 1,
              backgroundColor: beatIndex === 0 ? 'rgba(33, 150, 243, 0.1)' : 'transparent',
              borderRadius: 1
            }}
          >
            <Typography variant="body2" sx={{ minWidth: '60px' }}>
              Beat {beatIndex + 1}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flex: 1 }}>
              {Array(subdivision).fill(0).map((_, subdivIndex) => (
                <Box 
                  key={subdivIndex}
                  sx={{ 
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}
                >
                  <Slider
                    orientation="vertical"
                    value={accents[beatIndex][subdivIndex]}
                    onChange={(_, value) => handleAccentChange(beatIndex, subdivIndex, value)}
                    min={0}
                    max={100}
                    sx={{ height: 100 }}
                  />
                  <Typography variant="caption">
                    {subdivIndex + 1}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default BeatPatternEditor;
