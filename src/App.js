import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Button,
  Paper,
  ThemeProvider,
  createTheme,
  Tab,
  Tabs,
  Slider,
  Card,
  CardContent,
  Switch,
  CircularProgress,
  Alert
} from '@mui/material';
import SongAnalysisViewer from './components/SongAnalysisViewer';
import LyricsDisplay from './components/LyricsDisplay';
import TablatureDisplay from './components/TablatureDisplay';
import GuitarFretboard from './components/GuitarFretboard';
import apiService from './apiService';
import { 
  CloudUpload, 
  PlayArrow, 
  Pause, 
  FastForward, 
  FastRewind 
} from '@mui/icons-material';

// Helper function to format time in MM:SS format
const formatTime = (time) => {
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: "'Inter', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    h2: {
      fontWeight: 700,
    }
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 8px 40px rgba(0,0,0,0.12)'
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600
        }
      }
    }
  }
});

const instruments = [
  { name: 'Vocals', color: '#FF6B6B' },
  { name: 'Backup Vocals', color: '#4ECDC4' },
  { name: 'Lead Guitar', color: '#45B7D1' },
  { name: 'Rhythm Guitar', color: '#96CEB4' },
  { name: 'Acoustic Guitar', color: '#FFEEAD' },
  { name: 'Bass', color: '#D4A5A5' },
  { name: 'Drums', color: '#9B59B6' },
  { name: 'Piano', color: '#3498DB' }
];

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef(null);
  const animationRef = useRef(null);

  // Audio context ref
  const audioContext = useRef(null);

  // Audio playback implementation
  useEffect(() => {
    if (audioFile && audioRef.current) {
      const fileURL = URL.createObjectURL(audioFile);
      audioRef.current.src = fileURL;
      
      // Function to update time during playback
      const updateTime = () => {
        if (audioRef.current) {
          setCurrentTime(audioRef.current.currentTime);
          animationRef.current = requestAnimationFrame(updateTime);
        }
      };
      
      // Add event listeners for better state management
      const audio = audioRef.current;
      const handlePlay = () => {
        setIsPlaying(true);
        animationRef.current = requestAnimationFrame(updateTime);
      };
      
      const handlePause = () => {
        setIsPlaying(false);
        cancelAnimationFrame(animationRef.current);
      };
      
      const handleEnded = () => {
        setIsPlaying(false);
        setCurrentTime(0);
        cancelAnimationFrame(animationRef.current);
      };
      
      const handleTimeUpdate = () => {
        setCurrentTime(audio.currentTime);
      };
      
      audio.addEventListener('play', handlePlay);
      audio.addEventListener('pause', handlePause);
      audio.addEventListener('ended', handleEnded);
      audio.addEventListener('timeupdate', handleTimeUpdate);
      
      return () => {
        URL.revokeObjectURL(fileURL);
        audio.removeEventListener('play', handlePlay);
        audio.removeEventListener('pause', handlePause);
        audio.removeEventListener('ended', handleEnded);
        audio.removeEventListener('timeupdate', handleTimeUpdate);
        cancelAnimationFrame(animationRef.current);
      };
    }
  }, [audioFile]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    console.log('File selected:', file.name, file.type, `${Math.round(file.size/1024)} KB`);

    // Validate file type and size
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/x-wav'];
    if (!validTypes.includes(file.type) && !file.name.endsWith('.mp3') && !file.name.endsWith('.wav')) {
      setError(`Please upload a valid audio file (MP3 or WAV). Detected type: ${file.type}`);
      return;
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      setError('File size must be less than 100MB');
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      console.log('Uploading file to server...');
      const response = await apiService.uploadAudio(file);
      console.log('Upload successful:', response);

      setAudioFile(file);
      setIsPlaying(false);
      
      // First load the file locally for preview
      const localUrl = URL.createObjectURL(file);
      audioRef.current.src = localUrl;
      
      // Then set the backend URL for analysis
      const backendUrl = apiService.playAudio(response.filename);
      console.log('Backend audio URL:', backendUrl);
      
      // Pre-fetch some analysis to ensure connection works
      try {
        const keyData = await apiService.getKey(response.filename);
        console.log('Key detection successful:', keyData);
      } catch (analysisError) {
        console.warn('Initial analysis check failed:', analysisError);
      }
      
    } catch (error) {
      console.error('Upload error details:', error.response?.data || error.message || error);
      setError(
        `Failed to upload audio file: ${error.response?.data?.error || error.message || 'Unknown error'}. Please ensure the backend server is running at ${apiService.getBaseUrl()}.`
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const [filteredTracks, setFilteredTracks] = useState(instruments);
  const [searchTerm, setSearchTerm] = useState('');

  const InstrumentTrack = ({ instrument }) => {
    const [volume, setVolume] = useState(100);
    const [muted, setMuted] = useState(false);

    const handleVolumeChange = (e, value) => {
      setVolume(value);
      // Update audio context with new volume
      if (audioContext.current) {
        const gainNode = audioContext.current.createGain();
        gainNode.gain.value = muted ? 0 : value / 100;
        // Connect gain node to appropriate audio source
      }
    };

    return (
      <Card sx={{ mb: 2, borderRadius: 3, boxShadow: '0 4px 15px rgba(0,0,0,0.05)' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="h6" sx={{ color: instrument.color }}>
                {instrument.name}
              </Typography>
            </Box>
            <Switch
              checked={!muted}
              onChange={() => setMuted(!muted)}
              color="primary"
            />
            <Slider
              value={volume}
              onChange={handleVolumeChange}
              min={0}
              max={100}
              sx={{ width: 100, ml: 2 }}
              disabled={muted}
            />
          </Box>
        </CardContent>
      </Card>
    );
  };

  // State for storing tablature analysis data
  const [tablatureData, setTablatureData] = useState(null);
  const [notePositions, setNotePositions] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Function to fetch tablature data from the backend
  const fetchTablatureData = useCallback(async (filename) => {
    if (!filename) return;
    
    try {
      setIsAnalyzing(true);
      const secureFilename = encodeURIComponent(filename);
      console.log(`Fetching tablature data for: ${secureFilename}`);
      
      // Get the main tablature structure analysis
      const structureResponse = await apiService.getStructure(secureFilename);
      setTablatureData(structureResponse);
      
      // Get the fretboard position data
      const fretboardResponse = await apiService.getFretboardPositions(secureFilename);
      setNotePositions(fretboardResponse.notePositions || []);
      
      console.log('Tablature analysis complete', structureResponse);
    } catch (error) {
      console.error('Error fetching tablature data:', error);
      setError(`Unable to analyze tablature: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [setError]);

  const handleSearch = (e) => {
    const term = e.target.value.toLowerCase();
    setSearchTerm(term);
    const filtered = instruments.filter(instrument => 
      instrument.name.toLowerCase().includes(term)
    );
    setFilteredTracks(filtered);
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg">
        {/* Add hidden audio element */}
        <audio ref={audioRef} style={{ display: 'none' }} />
        
        <Box sx={{ my: 4 }}>
          <Typography variant="h2" component="h1" gutterBottom align="center" sx={{ 
            fontWeight: 'bold',
            background: 'linear-gradient(45deg, #2196f3 30%, #21CBF3 90%)',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            mb: 4
          }}>
            OliTunes
          </Typography>
          
          <Paper elevation={3} sx={{ p: 4, borderRadius: 3, mb: 4 }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <input
                accept="audio/*"
                style={{ display: 'none' }}
                id="audio-file-upload"
                type="file"
                onChange={handleFileUpload}
              />
              <label htmlFor="audio-file-upload">
                <Button
                  variant="contained"
                  component="span"
                  startIcon={<CloudUpload />}
                  sx={{ 
                    mb: 3,
                    py: 1.5,
                    background: 'linear-gradient(45deg, #2196f3 30%, #21CBF3 90%)',
                    boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)'
                  }}
                >
                  Upload Audio File
                </Button>
              </label>
            </Box>

            {error && (
              <Box sx={{ mb: 3 }}>
                <Alert severity="error" onClose={() => setError(null)}>
                  {error}
                </Alert>
              </Box>
            )}

            {isLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
              </Box>
            ) : audioFile && (
              <>
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'center', 
                  gap: 2,
                  mb: 4
                }}>
                  <Button 
                    variant="contained" 
                    startIcon={<FastRewind />}
                    onClick={() => audioRef.current.currentTime -= 5}
                    sx={{ 
                      minWidth: '120px',
                      background: 'linear-gradient(45deg, #3f51b5 30%, #5c6bc0 90%)',
                    }}
                  >
                    Rewind
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={isPlaying ? <Pause /> : <PlayArrow />}
                    onClick={togglePlayback}
                    sx={{ 
                      minWidth: '120px',
                      background: isPlaying 
                        ? 'linear-gradient(45deg, #f44336 30%, #e57373 90%)'
                        : 'linear-gradient(45deg, #4caf50 30%, #81c784 90%)',
                    }}
                  >
                    {isPlaying ? 'Pause' : 'Play'}
                  </Button>
                  <Button 
                    variant="contained" 
                    startIcon={<FastForward />}
                    onClick={() => audioRef.current.currentTime += 5}
                    sx={{ 
                      minWidth: '120px',
                      background: 'linear-gradient(45deg, #3f51b5 30%, #5c6bc0 90%)',
                    }}
                  >
                    Forward
                  </Button>
                </Box>

                {/* Audio playback time information */}
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  gap: 2,
                  mb: 3,
                  p: 2,
                  borderRadius: 4,
                  bgcolor: 'rgba(33,150,243,0.05)',
                  border: '1px solid rgba(33,150,243,0.1)',
                }}>
                  <Typography variant="body2" sx={{ 
                    fontWeight: 600,
                    fontFamily: 'monospace',
                    minWidth: '45px'
                  }}>
                    {formatTime(currentTime)}
                  </Typography>
                  
                  <Slider
                    value={currentTime}
                    min={0}
                    max={audioRef.current?.duration || 100}
                    onChange={(e, newValue) => {
                      if (audioRef.current) {
                        audioRef.current.currentTime = newValue;
                        setCurrentTime(newValue);
                      }
                    }}
                    sx={{ flexGrow: 1, mx: 2, maxWidth: '500px' }}
                  />
                  
                  <Typography variant="body2" sx={{ 
                    fontWeight: 600,
                    fontFamily: 'monospace',
                    minWidth: '45px'
                  }}>
                    {formatTime(audioRef.current?.duration || 0)}
                  </Typography>
                </Box>

                <Tabs 
                  value={currentTab} 
                  onChange={handleTabChange} 
                  centered
                  sx={{ 
                    borderBottom: 1, 
                    borderColor: 'divider', 
                    mb: 3,
                    '& .MuiTabs-indicator': {
                      height: 3,
                      borderRadius: 1.5
                    }
                  }}
                >
                  <Tab label="Tablature" />
                  <Tab label="Lyrics" />
                  <Tab label="Tracks" />
                  <Tab label="Structure" />
                </Tabs>

                <Box sx={{ mt: 2 }}>
                  {currentTab === 0 ? (
                    <Box>
                      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                        Guitar Analysis & Tablature
                      </Typography>
                      
                      {!tablatureData && !isAnalyzing && audioFile && (
                        <Box sx={{ textAlign: 'center', my: 4 }}>
                          <Button 
                            variant="contained" 
                            color="primary"
                            onClick={() => fetchTablatureData(audioFile.name)}
                            sx={{
                              py: 1.5,
                              px: 3,
                              background: 'linear-gradient(45deg, #2196f3 30%, #21CBF3 90%)',
                            }}
                          >
                            Analyze and Generate Tablature
                          </Button>
                          <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>
                            This will analyze the audio file and generate guitar tablature with enhanced pitch detection.
                          </Typography>
                        </Box>
                      )}
                      
                      {isAnalyzing && (
                        <Box sx={{ textAlign: 'center', my: 4 }}>
                          <CircularProgress />
                          <Typography variant="body1" sx={{ mt: 2 }}>
                            Analyzing audio and generating intelligent tablature...
                          </Typography>
                          <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary' }}>
                            This may take a few moments depending on the file length.
                          </Typography>
                        </Box>
                      )}
                      
                      {tablatureData && (
                        <>
                          <GuitarFretboard 
                            notes={tablatureData.notes || []}
                            currentTime={currentTime}
                            analysisData={tablatureData}
                          />
                          
                          <TablatureDisplay 
                            tablature={tablatureData}
                            currentTime={currentTime}
                            notePositions={notePositions}
                          />
                        </>
                      )}
                    </Box>
                  ) : currentTab === 1 ? (
                    <LyricsDisplay filename={audioFile?.name} currentTime={currentTime} />
                  ) : currentTab === 2 ? (
                    <Box>
                      <Box sx={{ mb: 3 }}>
                        <input
                          type="text"
                          value={searchTerm}
                          onChange={handleSearch}
                          placeholder="Search tracks..."
                          style={{
                            width: '100%',
                            padding: '12px',
                            fontSize: '16px',
                            border: '1px solid #ddd',
                            borderRadius: '8px'
                          }}
                        />
                      </Box>
                      {filteredTracks.map((instrument) => (
                        <InstrumentTrack key={instrument.name} instrument={instrument} />
                      ))}
                    </Box>
                  ) : (
                    <SongAnalysisViewer filename={audioFile?.name} currentTime={currentTime} />
                  )}
                </Box>
              </>
            )}
          </Paper>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
