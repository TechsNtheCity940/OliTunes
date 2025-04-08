import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Button,
  Grid,
  Card,
  CardContent,
  CardMedia,
  TextField,
  IconButton,
  Switch,
  CircularProgress,
  Divider,
  Slider,
  LinearProgress,
  Paper,
  ThemeProvider,
  Tab,
  Tabs,
  CardActions,
  Alert,
  CssBaseline
} from '@mui/material';
import { 
  CloudUpload, 
  PlayArrow, 
  Pause, 
  FastForward, 
  FastRewind,
  MusicNote,
  LibraryMusic,
  GraphicEq,
  Assessment,
  PendingActions,
  Piano as PianoIcon,
  Feedback as FeedbackIcon,
  BarChart,
  Search,
  Download
} from '@mui/icons-material';

// Import components
import SongAnalysisViewer from '../../frontend/src/components/SongAnalysisViewer';
import LyricsDisplay from '../../frontend/src/components/LyricsDisplay';
import TablatureDisplay from '../../frontend/src/components/TablatureDisplay';
import GuitarFretboard from '../../frontend/src/components/GuitarFretboard';
import PianoKeyboard from '../../frontend/src/components/PianoKeyboard';
import Pendulum from '../../frontend/src/components/Pendulum';
import ModelPerformance from '../../frontend/src/components/ModelPerformance';
import FeedbackForm from '../../frontend/src/components/FeedbackForm';
import FeedbackStats from '../../frontend/src/components/FeedbackStats';
import BeatPatternEditor from '../../frontend/src/components/BeatPatternEditor';
import TabManager from '../../frontend/src/components/TabManager';
import TextTablature from '../../frontend/src/components/TextTablature';
import apiService from '../../frontend/src/apiService';
import '../../frontend/src/components/custom-theme.css';
import musicTheme from '../../frontend/src/theme/musicTheme';

// Helper function to format time in MM:SS format
const formatTime = (time) => {
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

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
  const [animateNotes, setAnimateNotes] = useState(true);
  const [volume, setVolume] = useState(100);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef(null);
  const animationRef = useRef(null);

  // State for new components
  const [modelPerformanceData, setModelPerformanceData] = useState(null);
  const [feedbackData, setFeedbackData] = useState({ rating: 0, comments: '' });
  const [tabSearchQuery, setTabSearchQuery] = useState('');
  const [isTabSearching, setIsTabSearching] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [downloadJobs, setDownloadJobs] = useState([]);
  const [showTabManager, setShowTabManager] = useState(false);
  
  // Beat pattern editor state
  const [beatPattern, setBeatPattern] = useState([
    { duration: '1/4', velocity: 0.8, isRest: false },
    { duration: '1/4', velocity: 0.5, isRest: false },
    { duration: '1/4', velocity: 0.6, isRest: false },
    { duration: '1/4', velocity: 0.5, isRest: false }
  ]);
  const [subdivision, setSubdivision] = useState(2);
  const [accents, setAccents] = useState([
    [80, 40, 60, 30],
    [60, 30, 40, 20],
    [70, 35, 50, 25],
    [50, 25, 30, 15]
  ]);

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
      
      const handleLoadedMetadata = () => {
        setDuration(audio.duration);
      };
      
      audio.addEventListener('play', handlePlay);
      audio.addEventListener('pause', handlePause);
      audio.addEventListener('ended', handleEnded);
      audio.addEventListener('timeupdate', handleTimeUpdate);
      audio.addEventListener('loadedmetadata', handleLoadedMetadata);
      
      return () => {
        URL.revokeObjectURL(fileURL);
        audio.removeEventListener('play', handlePlay);
        audio.removeEventListener('pause', handlePause);
        audio.removeEventListener('ended', handleEnded);
        audio.removeEventListener('timeupdate', handleTimeUpdate);
        audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
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
  const [tabGenerated, setTabGenerated] = useState(false);
  const [textTablature, setTextTablature] = useState(null);
  const [tabError, setTabError] = useState(null);
  
  // Function to fetch tablature data from the backend
  const fetchTablatureData = useCallback(async (filename) => {
    if (!filename) return;
    
    try {
      setIsAnalyzing(true);
      setTabError(null);
      const secureFilename = encodeURIComponent(filename);
      console.log(`Fetching tablature data for: ${secureFilename}`);
      
      // Get the main tablature structure analysis
      try {
        const structureResponse = await apiService.getStructure(secureFilename);
        setTablatureData(structureResponse);
      } catch (err) {
        console.warn('Error fetching structure data:', err);
      }
      
      // Get the fretboard position data
      try {
        const fretboardResponse = await apiService.getFretboardPositions(secureFilename);
        setNotePositions(fretboardResponse.notePositions || []);
      } catch (err) {
        console.warn('Error fetching fretboard positions:', err);
      }
      
      // Get the text-based tablature
      try {
        const textTabResponse = await apiService.getTextTablature(secureFilename);
        setTextTablature(textTabResponse.tablature);
        console.log('Text tablature received:', textTabResponse);
      } catch (err) {
        console.error('Error fetching text tablature:', err);
        setTabError(`Unable to generate text tablature: ${err.message}`);
      }
      
      // Fetch model performance data
      try {
        const performanceData = await apiService.getModelPerformance();
        setModelPerformanceData(performanceData);
      } catch (err) {
        console.warn('Error fetching model performance data:', err);
      }
      
      console.log('Tablature analysis complete');
      setTabGenerated(true);
    } catch (error) {
      console.error('Error fetching tablature data:', error);
      setTabError(`Unable to analyze tablature: ${error.message}`);
      setTabGenerated(false);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  useEffect(() => {
    if (audioFile && !tabGenerated) {
      fetchTablatureData(audioFile.name);
      setTabGenerated(true);
    }
  }, [audioFile, tabGenerated, fetchTablatureData]);

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

  const handlePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleStop = () => {
    if (!audioRef.current) return;

    audioRef.current.pause();
    audioRef.current.currentTime = 0;
    setIsPlaying(false);
  };

  const skipTime = (seconds) => {
    if (!audioRef.current) return;

    audioRef.current.currentTime += seconds;
  };

  const handleSliderChange = (e, newValue) => {
    if (audioRef.current) {
      audioRef.current.currentTime = newValue;
      setCurrentTime(newValue);
    }
  };

  const handleVolumeChange = (e, newValue) => {
    setVolume(newValue);
  };

  // Function to search for tabs
  const searchForTabs = async () => {
    if (!tabSearchQuery.trim()) return;
    
    setIsTabSearching(true);
    try {
      const response = await apiService.searchTabs(tabSearchQuery);
      setSearchResults(response.results || []);
    } catch (err) {
      console.error('Error searching for tabs:', err);
      setError(`Error searching for tabs: ${err.message}`);
    } finally {
      setIsTabSearching(false);
    }
  };
  
  // Function to download a tab
  const downloadTab = async (tabId, artist, title) => {
    try {
      const response = await apiService.downloadTab(tabId, artist, title);
      // Add to jobs list
      setDownloadJobs(prev => [...prev, {
        id: response.jobId,
        artist,
        title,
        status: 'downloading',
        progress: 0
      }]);
      
      // Poll for job status
      pollJobStatus(response.jobId);
    } catch (err) {
      console.error('Error downloading tab:', err);
      setError(`Error downloading tab: ${err.message}`);
    }
  };
  
  // Function to poll job status
  const pollJobStatus = async (jobId) => {
    const checkStatus = async () => {
      try {
        const status = await apiService.checkJobStatus(jobId);
        setDownloadJobs(prev => prev.map(job => 
          job.id === jobId ? { ...job, status: status.status, progress: status.progress || 0 } : job
        ));
        
        if (status.status !== 'completed' && status.status !== 'failed') {
          setTimeout(checkStatus, 2000);
        }
      } catch (err) {
        console.error('Error checking job status:', err);
      }
    };
    
    checkStatus();
  };
  
  // Function to submit feedback
  const submitFeedback = async () => {
    try {
      await apiService.submitFeedback(feedbackData);
      // Reset form
      setFeedbackData({ rating: 0, comments: '' });
      // Show success message
      setError({ message: 'Feedback submitted successfully!', severity: 'success' });
    } catch (err) {
      console.error('Error submitting feedback:', err);
      setError(`Error submitting feedback: ${err.message}`);
    }
  };

  return (
    <ThemeProvider theme={musicTheme}>
      <CssBaseline />
      <audio ref={audioRef} style={{ display: 'none' }} />
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
          <Typography variant="h4" component="h1" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
            OliTunes <span style={{ fontSize: '0.7em', color: '#673ab7' }}>AI</span>
          </Typography>
          <Button 
            variant="contained" 
            component="label"
            startIcon={<CloudUpload />}
            disabled={isLoading}
          >
            Upload Song
            <input
              type="file"
              hidden
              accept=".mp3,.wav"
              onChange={handleFileUpload}
              disabled={isLoading}
            />
          </Button>
        </Box>
        
        {error && (
          <Alert 
            severity={error.severity || "error"} 
            sx={{ mb: 3 }}
            onClose={() => setError(null)}
          >
            {error.message || error}
          </Alert>
        )}
        
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <CircularProgress />
          </Box>
        )}
        
        <Paper sx={{ mb: 4 }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab icon={<PlayArrow />} label="Player" />
            <Tab icon={<MusicNote />} label="Analysis" />
            <Tab icon={<GraphicEq />} label="Mixer" />
            <Tab icon={<MusicNote />} label="Tablature" />
            <Tab icon={<PianoIcon />} label="Keyboard" />
            <Tab icon={<PendingActions />} label="Pendulum" />
            <Tab icon={<Assessment />} label="Model Performance" />
            <Tab icon={<FeedbackIcon />} label="Feedback" />
            <Tab icon={<BarChart />} label="Stats" />
            <Tab icon={<Download />} label="Tab Manager" />
          </Tabs>
          
          {/* Player Tab */}
          {currentTab === 0 && (
            <Box sx={{ p: 3 }}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="h5" gutterBottom>
                  {audioFile ? audioFile.name.replace(/\.[^/.]+$/, "") : "No file selected"}
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <IconButton onClick={handlePlayPause} disabled={!audioFile || isLoading}>
                    {isPlaying ? <Pause /> : <PlayArrow />}
                  </IconButton>
                  <IconButton onClick={handleStop} disabled={!audioFile || isLoading}>
                    <FastRewind />
                  </IconButton>
                  <IconButton onClick={() => skipTime(10)} disabled={!audioFile || isLoading}>
                    <FastForward />
                  </IconButton>
                  <Box sx={{ width: '100%', ml: 2 }}>
                    <Slider
                      min={0}
                      max={duration}
                      value={currentTime}
                      onChange={handleSliderChange}
                      valueLabelDisplay="auto"
                      valueLabelFormat={formatTime}
                      disabled={!audioFile || isLoading}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption">{formatTime(currentTime)}</Typography>
                      <Typography variant="caption">{formatTime(duration)}</Typography>
                    </Box>
                  </Box>
                </Box>
              </Box>
              
              {/* Visualization goes here */}
            </Box>
          )}
          
          {/* Analysis Tab */}
          {currentTab === 1 && (
            <Box sx={{ p: 3 }}>
              <SongAnalysisViewer />
            </Box>
          )}
          
          {/* Mixer Tab */}
          {currentTab === 2 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Instrument Mixer
              </Typography>
              <Box>
                {filteredTracks.map((instrument, index) => (
                  <InstrumentTrack key={index} instrument={instrument} />
                ))}
              </Box>
            </Box>
          )}
          
          {/* Tablature Tab */}
          {currentTab === 3 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Guitar Tablature
              </Typography>
              
              {!audioFile && (
                <Alert severity="info" sx={{ mb: 3 }}>
                  Please upload an audio file first to generate a tablature.
                </Alert>
              )}
              
              {audioFile && !tabGenerated && !isAnalyzing && (
                <Box sx={{ mb: 3 }}>
                  <Button 
                    variant="contained" 
                    color="primary" 
                    onClick={() => {
                      if (audioFile) {
                        fetchTablatureData(audioFile.name);
                        setTabGenerated(true);
                      }
                    }}
                    startIcon={<MusicNote />}
                  >
                    Analyze Audio and Generate Tabs
                  </Button>
                </Box>
              )}
              
              {isAnalyzing && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 4 }}>
                  <CircularProgress />
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Analyzing audio and generating tablature...
                  </Typography>
                </Box>
              )}
              
              {/* Display the text-based tablature */}
              {!isAnalyzing && audioFile && (
                <TextTablature 
                  tablature={textTablature} 
                  isLoading={isAnalyzing} 
                  error={tabError} 
                />
              )}
            </Box>
          )}
          
          {/* Piano Keyboard Tab */}
          {currentTab === 4 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Piano Keyboard
              </Typography>
              <PianoKeyboard 
                notePositions={notePositions} 
                currentTime={currentTime} 
                isPlaying={isPlaying}
              />
            </Box>
          )}
          
          {/* Pendulum Tab */}
          {currentTab === 5 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Tempo Visualization
              </Typography>
              <Pendulum 
                tempo={tablatureData?.tempo || 120} 
                isPlaying={isPlaying}
              />
            </Box>
          )}
          
          {/* Model Performance Tab */}
          {currentTab === 6 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Model Performance Metrics
              </Typography>
              <ModelPerformance data={modelPerformanceData} />
            </Box>
          )}
          
          {/* Feedback Tab */}
          {currentTab === 7 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Provide Feedback
              </Typography>
              <FeedbackForm 
                value={feedbackData}
                onChange={setFeedbackData}
                onSubmit={submitFeedback}
              />
            </Box>
          )}
          
          {/* Stats Tab */}
          {currentTab === 8 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Feedback Statistics
              </Typography>
              <FeedbackStats />
            </Box>
          )}
          
          {/* Tab Manager Tab */}
          {currentTab === 9 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Tab Manager
              </Typography>
              <TabManager />
            </Box>
          )}
        </Paper>
      </Container>
    </ThemeProvider>
  );
}

export default App;
