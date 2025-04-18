import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  Chip,
  List,
  ListItem,
  ListItemText,
  Card,
  CardContent,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  useTheme
} from '@mui/material';
import MusicNoteIcon from '@mui/icons-material/MusicNote';
import AudiotrackIcon from '@mui/icons-material/Audiotrack';
import PianoIcon from '@mui/icons-material/Piano';
import LibraryMusicIcon from '@mui/icons-material/LibraryMusic';
import PianoKeyboard from './PianoKeyboard';
import apiService from '../apiService';

const SongAnalysisViewer = ({ filename, currentTime }) => {
  const theme = useTheme();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [currentBar, setCurrentBar] = useState(null);
  const [currentMeasure, setCurrentMeasure] = useState(null);

  useEffect(() => {
    if (!filename) return;
    
    const fetchAnalysis = async () => {
      try {
        setLoading(true);
        const data = await apiService.getStructureAnalysis(filename);
        setAnalysis(data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching song analysis:', err);
        setError('Failed to analyze song. Please try again.');
        setLoading(false);
      }
    };

    fetchAnalysis();
  }, [filename]);

  // Update current bar and measure based on playback time
  useEffect(() => {
    if (!analysis || !analysis.bars) return;

    // Find current bar
    const bar = analysis.bars.find(b => 
      currentTime >= b.startTime && currentTime < b.endTime
    );

    if (bar) {
      setCurrentBar(bar);
      
      // Find current measure within the bar
      const measure = bar.measures.find(m =>
        currentTime >= m.startTime && currentTime < m.endTime
      );
      
      if (measure) {
        setCurrentMeasure(measure);
      }
    }
  }, [currentTime, analysis]);

  // Format time (seconds) to MM:SS format
  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Format chord with appropriate color based on its type
  const getChordColor = (chord) => {
    if (chord.includes('m') && !chord.includes('maj')) {
      return '#e57373'; // minor chords - reddish
    } else if (chord.includes('7')) {
      return '#5c6bc0'; // 7th chords - bluish
    } else if (chord.includes('dim')) {
      return '#9575cd'; // diminished - purple
    } else if (chord.includes('aug')) {
      return '#4db6ac'; // augmented - teal
    } else if (chord.includes('sus')) {
      return '#ffb74d'; // suspended - orange
    } else {
      return '#66bb6a'; // major chords - green
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress color="secondary" />
        <Typography sx={{ ml: 2, color: '#555' }}>
          Analyzing audio and extracting musical information...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" variant="filled">{error}</Alert>
    );
  }

  if (!analysis) {
    return (
      <Typography align="center" my={4}>
        Upload and select an audio file to analyze.
      </Typography>
    );
  }

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        height: '100%', 
        overflow: 'auto',
        position: 'relative',
        background: 'linear-gradient(to bottom, #f9f9f9, #ffffff)',
        backgroundImage: 'url(https://www.transparenttextures.com/patterns/music-pattern.png)',
        backgroundBlendMode: 'overlay',
        border: '1px solid #e0e0e0',
        borderRadius: '8px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.08)'
      }}
    >
      <Box sx={{ mt: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" sx={{ 
            fontWeight: 'bold', 
            color: theme.palette.primary.main,
            display: 'flex',
            alignItems: 'center'
          }}>
            <LibraryMusicIcon sx={{ mr: 1 }} />
            Music Analysis Results
          </Typography>
          
          <Chip 
            icon={<AudiotrackIcon />} 
            label={filename ? filename.replace(/_/g, ' ').replace(/\.[^/.]+$/, '') : ''}
            variant="outlined"
            color="secondary"
          />
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Musical Information
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  <Chip 
                    label={`Key: ${analysis.key}`} 
                    color="primary" 
                    variant="outlined"
                  />
                  <Chip 
                    label={`Tempo: ${Math.round(analysis.tempo)} BPM`} 
                    color="primary" 
                    variant="outlined"
                  />
                  <Chip 
                    label={`Time Signature: ${analysis.timeSignature.numerator}/${analysis.timeSignature.denominator}`} 
                    color="primary" 
                    variant="outlined"
                  />
                  <Chip 
                    label={`Duration: ${formatTime(analysis.totalDuration)}`} 
                    color="primary" 
                    variant="outlined"
                  />
                </Box>
                
                <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                  Current Position
                </Typography>
                {currentBar && currentMeasure ? (
                  <Box>
                    <Typography variant="body1">
                      Bar {currentBar.barNumber}, Measure {currentMeasure.measureNumber}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {formatTime(currentTime)} / {formatTime(analysis.totalDuration)}
                    </Typography>
                  </Box>
                ) : (
                  <Typography variant="body2" color="textSecondary">
                    Play the audio to see real-time position
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Chord Progression
                </Typography>
                <Box sx={{ 
                  display: 'flex', 
                  flexWrap: 'wrap', 
                  gap: 1.5, 
                  mb: 2,
                  maxHeight: '200px',
                  overflowY: 'auto',
                  p: 1
                }}>
                  {analysis.chordProgression && analysis.chordProgression.map((chordData, index) => {
                    const isCurrentChord = currentTime >= chordData.start_time && 
                      currentTime < (chordData.start_time + chordData.duration);
                      
                    return (
                      <Chip 
                        key={index}
                        label={chordData.chord} 
                        sx={{ 
                          bgcolor: getChordColor(chordData.chord),
                          color: 'white',
                          fontWeight: isCurrentChord ? 'bold' : 'normal',
                          transform: isCurrentChord ? 'scale(1.1)' : 'scale(1)',
                          transition: 'all 0.2s ease-in-out',
                          border: isCurrentChord ? '2px solid #fff' : 'none',
                          boxShadow: isCurrentChord ? '0 0 10px rgba(0,0,0,0.3)' : 'none'
                        }}
                        variant={isCurrentChord ? "filled" : "outlined"}
                      />
                    );
                  })}
                </Box>
                
                <Typography variant="subtitle2" color="textSecondary" sx={{ mt: 2 }}>
                  Chord detection confidence: {
                    analysis.chordProgression?.length > 0 
                      ? `${Math.round(analysis.chordProgression[0].confidence * 100)}%` 
                      : 'N/A'
                  }
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
      
      <Box sx={{ mt: 4 }}>
        <Tabs 
          value={activeTab} 
          onChange={(e, newValue) => setActiveTab(newValue)}
          variant="fullWidth"
          textColor="secondary"
          indicatorColor="secondary"
          sx={{ 
            mb: 2, 
            borderBottom: 1, 
            borderColor: 'divider',
            '& .MuiTab-root': {
              fontWeight: 'bold',
              borderRadius: '4px 4px 0 0',
              transition: 'all 0.2s',
              '&:hover': {
                backgroundColor: 'rgba(103, 58, 183, 0.04)',
              },
            }
          }}
        >
          <Tab icon={<MusicNoteIcon />} label="Tablature" />
          <Tab icon={<AudiotrackIcon />} label="Notes" />
          <Tab icon={<PianoIcon />} label="Piano Keyboard" />
        </Tabs>
        
        <TabPanel value={activeTab} index={0}>
          <Box sx={{ maxHeight: '400px', overflowY: 'auto', px: 2 }}>
            {analysis.bars?.map((bar, barIndex) => (
              <Box 
                key={barIndex}
                sx={{ 
                  mt: 3,
                  p: 2,
                  backgroundColor: currentBar?.barNumber === bar.barNumber ? '#f0f7ff' : 'transparent',
                  borderRadius: 1,
                  border: '1px solid #e0e0e0'
                }}
              >
                <Typography variant="subtitle1" gutterBottom>
                  Bar {bar.barNumber} ({formatTime(bar.startTime)} - {formatTime(bar.endTime)})
                </Typography>
                
                <Grid container spacing={2}>
                  {bar.measures?.map((measure, measureIndex) => (
                    <Grid item xs={12} md={6} lg={3} key={measureIndex}>
                      <Paper
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
                        elevation={1}
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
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            ))}
          </Box>
        </TabPanel>
        
        <TabPanel value={activeTab} index={1}>
          <Box sx={{ maxHeight: '400px', overflowY: 'auto' }}>
            <List dense>
              {analysis.notes?.map((note, index) => (
                <ListItem 
                  key={index}
                  sx={{
                    backgroundColor: 
                      currentTime >= note.time && 
                      currentTime < (note.time + note.duration) 
                        ? 'rgba(33, 150, 243, 0.1)'
                        : 'transparent',
                    borderRadius: 1
                  }}
                >
                  <ListItemText
                    primary={
                      <Typography variant="body1">
                        <strong>Note:</strong> {note.note} 
                        {note.confidence && ` (Confidence: ${Math.round(note.confidence * 100)}%)`}
                      </Typography>
                    }
                    secondary={
                      <Typography variant="body2" color="textSecondary">
                        Time: {formatTime(note.time)} - Duration: {note.duration.toFixed(2)}s
                      </Typography>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        </TabPanel>
        
        <TabPanel value={activeTab} index={2}>
          <Box sx={{ maxHeight: '400px', overflowY: 'auto' }}>
            <PianoKeyboard 
              notes={analysis?.notes || []}
              currentTime={currentTime}
              octaveRange={{ min: 2, max: 6 }}
            />
          </Box>
        </TabPanel>
      </Box>
    </Paper>
  );
};

// TabPanel helper component
function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default SongAnalysisViewer;
