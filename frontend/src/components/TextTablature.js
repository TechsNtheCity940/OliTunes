import React from 'react';
import { Box, Typography, Paper, CircularProgress, Alert } from '@mui/material';

/**
 * Component for displaying text-based guitar tablature in monospace font
 */
const TextTablature = ({ tablature, isLoading, error }) => {
  // Process the tablature to properly handle the [tab] tags
  const processTabContent = (content) => {
    if (!content) return '';
    
    // Replace [tab] and [/tab] tags with appropriate React/HTML tags
    return content
      .replace(/\[tab\]/g, '')
      .replace(/\[\/tab\]/g, '');
  };

  return (
    <Box sx={{ width: '100%' }}>
      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}
      
      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          {error}
        </Alert>
      )}
      
      {!isLoading && !error && tablature && (
        <Paper 
          elevation={3} 
          sx={{ 
            p: 2, 
            overflowX: 'auto',
            backgroundColor: '#f9f9f9', 
            borderRadius: 2,
            my: 2
          }}
        >
          <Typography
            component="pre"
            sx={{
              fontFamily: '"Roboto Mono", monospace',
              fontSize: '14px',
              lineHeight: 1.5,
              whiteSpace: 'pre-wrap',
              overflowX: 'auto',
              '& span.chord': {
                fontWeight: 'bold',
                color: '#3f51b5'
              }
            }}
          >
            {processTabContent(tablature)}
          </Typography>
        </Paper>
      )}
      
      {!isLoading && !error && !tablature && (
        <Alert severity="info" sx={{ my: 2 }}>
          No tablature available. Upload an audio file and click "Generate Tablature" to create tabs.
        </Alert>
      )}
    </Box>
  );
};

export default TextTablature;
