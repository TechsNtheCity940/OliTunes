import React, { useState, useEffect, useRef } from 'react';
import apiService from '../apiService';
import { 
  CircularProgress, 
  Typography, 
  Box, 
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';

const LyricsDisplay = ({ filename, currentTime }) => {
  const [lyrics, setLyrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
  const lyricsRef = useRef(null);

  useEffect(() => {
    const fetchLyrics = async () => {
      if (!filename) return;
      
      try {
        setLoading(true);
        const data = await apiService.getLyrics(filename);
        setLyrics(data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching lyrics:', err);
        setError('Failed to load lyrics. Please try again.');
        setLoading(false);
      }
    };

    fetchLyrics();
  }, [filename]);

  // Update current segment based on playback time
  useEffect(() => {
    if (!lyrics || !lyrics.segments || !lyrics.segments.length) return;

    // Find the current segment based on the current playback time
    const index = lyrics.segments.findIndex(
      segment => currentTime >= segment.start && currentTime < segment.end
    );

    // If a valid segment found, update the current segment index
    if (index !== -1 && index !== currentSegmentIndex) {
      setCurrentSegmentIndex(index);
      
      // Scroll to the current segment
      if (lyricsRef.current) {
        const listItems = lyricsRef.current.querySelectorAll('li');
        if (listItems[index]) {
          listItems[index].scrollIntoView({
            behavior: 'smooth',
            block: 'center'
          });
        }
      }
    }
  }, [currentTime, lyrics, currentSegmentIndex]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" my={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" align="center" my={4}>
        {error}
      </Typography>
    );
  }

  if (!lyrics || !lyrics.segments || lyrics.segments.length === 0) {
    return (
      <Typography align="center" my={4}>
        No lyrics available for this track.
      </Typography>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3, maxHeight: '400px', overflow: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        Lyrics
      </Typography>
      
      <Divider sx={{ mb: 2 }} />
      
      <List ref={lyricsRef}>
        {lyrics.segments.map((segment, index) => (
          <ListItem 
            key={index}
            sx={{
              backgroundColor: index === currentSegmentIndex ? 'rgba(33, 150, 243, 0.1)' : 'transparent',
              borderRadius: 1,
              transition: 'background-color 0.3s ease'
            }}
          >
            <ListItemText
              primary={
                <Typography 
                  variant="body1"
                  sx={{ 
                    fontWeight: index === currentSegmentIndex ? 'bold' : 'normal',
                    color: index === currentSegmentIndex ? 'primary.main' : 'text.primary'
                  }}
                >
                  {segment.text}
                </Typography>
              }
              secondary={`${formatTime(segment.start)} - ${formatTime(segment.end)}`}
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

// Helper function to format time in MM:SS format
const formatTime = (time) => {
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

export default LyricsDisplay;
