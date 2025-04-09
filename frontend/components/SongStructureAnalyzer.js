import React, { useState, useEffect, useRef } from 'react';
import apiService from '../../../src/apiService';
import { 
  CircularProgress, 
  Typography, 
  Box, 
  Paper, 
  List, 
  ListItem, 
  ListItemText, 
  Grid,
  IconButton,
  Slider,
  Button
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';

const SongStructureAnalyzer = ({ filename }) => {
  const [structure, setStructure] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const fetchStructure = async () => {
      try {
        const data = await apiService.getStructureAnalysis(filename);
        setStructure(data);
        setDuration(data.totalDuration);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchStructure();
    
    // Initialize audio with correct backend URL
    const audioUrl = `${process.env.REACT_APP_API_URL || 'http://localhost:5000'}/audio/${filename}`;
    audioRef.current = new Audio(audioUrl);
    audioRef.current.addEventListener('timeupdate', handleTimeUpdate);
    audioRef.current.addEventListener('ended', handleEnded);
    audioRef.current.addEventListener('error', (e) => {
      console.error('Audio error:', e);
      setError('Error loading audio file');
    });

    return () => {
      if (audioRef.current) {
        audioRef.current.removeEventListener('timeupdate', handleTimeUpdate);
        audioRef.current.removeEventListener('ended', handleEnded);
        audioRef.current.removeEventListener('error', () => {});
      }
      cancelAnimationFrame(animationRef.current);
    };
  }, [filename]);

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
    cancelAnimationFrame(animationRef.current);
  };

  const togglePlayPause = () => {
    if (isPlaying) {
      audioRef.current.pause();
      cancelAnimationFrame(animationRef.current);
    } else {
      audioRef.current.play();
      animationRef.current = requestAnimationFrame(updateTime);
    }
    setIsPlaying(!isPlaying);
  };

  const handleStop = () => {
    audioRef.current.pause();
    audioRef.current.currentTime = 0;
    setIsPlaying(false);
    setCurrentTime(0);
    cancelAnimationFrame(animationRef.current);
  };

  const updateTime = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      animationRef.current = requestAnimationFrame(updateTime);
    }
  };

  const handleSliderChange = (event, newValue) => {
    if (audioRef.current) {
      audioRef.current.currentTime = newValue;
      setCurrentTime(newValue);
    }
  };

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getCurrentBar = () => {
    if (!structure?.bars) return null;
    return structure.bars.find(bar => 
      currentTime >= bar.startTime && currentTime < bar.endTime
    );
  };

  const getCurrentMeasure = (bar) => {
    if (!bar?.measures) return null;
    return bar.measures.find(measure => 
      currentTime >= measure.startTime && currentTime < measure.endTime
    );
  };

  const renderTablature = () => {
    if (!structure?.bars || structure.bars.length === 0) {
      return <Typography>No tablature available</Typography>;
    }

    const currentBar = getCurrentBar();
    const currentMeasure = currentBar ? getCurrentMeasure(currentBar) : null;

    return (
      <Box>
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle1">
            Current Position: Bar {currentBar?.barNumber || 1}, 
            Measure {currentMeasure?.measureNumber || 1}
          </Typography>
        </Box>

        {structure.bars.map((bar, barIndex) => (
          <Box 
            key={barIndex}
            sx={{ 
              mt: 2,
              p: 2,
              backgroundColor: currentBar?.barNumber === bar.barNumber ? '#f0f7ff' : 'transparent',
              borderRadius: 1
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Bar {bar.barNumber} ({formatTime(bar.startTime)} - {formatTime(bar.endTime)})
            </Typography>
            
            <Grid container spacing={2}>
              {bar.measures.map((measure, measureIndex) => (
                <Grid item xs={12} md={3} key={measureIndex}>
                  <Box
                    sx={{
                      fontFamily: 'monospace',
                      whiteSpace: 'pre',
                      backgroundColor: currentMeasure?.measureNumber === measure.measureNumber &&
                                    currentBar?.barNumber === bar.barNumber ? '#e3f2fd' : '#f5f5f5',
                      p: 1,
                      borderRadius: 1,
                      fontSize: '14px',
                      lineHeight: 1.5,
                      letterSpacing: '0.1em'
                    }}
                  >
                    <Typography variant="caption" display="block" gutterBottom>
                      Measure {measure.measureNumber}
                    </Typography>
                    {measure.tabLines.map((line, lineIndex) => (
                      <Typography 
                        key={lineIndex}
                        sx={{ 
                          fontSize: 'inherit',
                          lineHeight: 'inherit',
                          letterSpacing: 'inherit'
                        }}
                      >
                        {line}
                      </Typography>
                    ))}
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        ))}
      </Box>
    );
  };

  const renderAudioControls = () => (
    <Box sx={{ mt: 2, mb: 4 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <IconButton onClick={togglePlayPause}>
          {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
        </IconButton>
        <IconButton onClick={handleStop}>
          <StopIcon />
        </IconButton>
        <Typography sx={{ ml: 2 }}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </Typography>
      </Box>
      <Slider
        value={currentTime}
        max={duration}
        onChange={handleSliderChange}
        aria-label="time-indicator"
      />
    </Box>
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" mt={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" align="center" mt={4}>
        Error: {error}
      </Typography>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        Song Structure Analysis
      </Typography>
      
      {renderAudioControls()}

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Musical Information
            </Typography>
            <Typography variant="body1">
              <strong>Key:</strong> {structure.key}
            </Typography>
            <Typography variant="body1">
              <strong>Time Signature:</strong> {structure.timeSignature.numerator}/{structure.timeSignature.denominator}
            </Typography>
          </Box>

          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Chord Progression
            </Typography>
            <List dense>
              {structure.chordProgression.map((chord, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={chord.chord}
                    secondary={`Time: ${chord.time.toFixed(2)}s`}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        </Grid>

        <Grid item xs={12} md={8}>
          <Typography variant="h6" gutterBottom>
            Guitar Tablature
          </Typography>
          {renderTablature()}
        </Grid>
      </Grid>
    </Paper>
  );
};

export default SongStructureAnalyzer;
