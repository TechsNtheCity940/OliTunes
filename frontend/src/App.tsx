import React, { useState, useRef, useEffect } from 'react';
import {
  Box, Button, Container, Typography, Paper, TextField, Grid,
  LinearProgress, Snackbar, Alert, Chip, Divider, IconButton,
  Drawer, List, ListItem, ListItemText, ListItemIcon, useTheme,
  CssBaseline, ThemeProvider, Tabs, Tab, CircularProgress,
  Backdrop, Tooltip, Slider, Stack as MuiStack, Card, CardContent, 
  CardMedia, Avatar, AppBar, Toolbar, Zoom, Fade, useMediaQuery
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  MusicNote as MusicNoteIcon,
  GraphicEq as GraphicEqIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Fullscreen as FullscreenIcon,
  ArrowBack as ArrowBackIcon,
  Tune as TuneIcon,
  Download as DownloadIcon,
  Settings as SettingsIcon,
  Close as CloseIcon,
  AudioFile as AudioFileIcon,
  Home as HomeIcon,
  Equalizer as EqualizerIcon,
  LibraryMusic as LibraryMusicIcon,
  ChevronRight as ChevronRightIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
  Info as InfoIcon,
  Explore as ExploreIcon,
  MusicVideo as MusicVideoIcon,
  Menu as MenuIcon
} from '@mui/icons-material';
import './App.css';

// Import all components from a single entry point
import { 
  GuitarFretboard, 
  TablatureDisplay, 
  PianoKeyboard, 
  SongAnalysisViewer,
  SongStructureAnalyzer,
  TablatureManager,
  TabManager,
  BeatPatternEditor,
  LyricsDisplay,
  TextTablature,
  ModelPerformance,
  FeedbackForm
} from './components';

// Import our music theme
import musicTheme from './theme/musicTheme';
import { getRandomQuote } from '../assets/musicQuotes';
import '../assets/animations.css';

// API service for backend communication
import apiService from './apiService';

// Use our imported theme
const darkTheme = musicTheme;

// Application constants
const API_URL = 'http://localhost:5000';

function App() {
  // State variables
  const [file, setFile] = useState<File | null>(null);
  const [audioInfo, setAudioInfo] = useState<{url: string, filename: string} | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [notesData, setNotesData] = useState<any>(null);
  const [tablatureData, setTablatureData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [notification, setNotification] = useState<{message: string, severity: 'success' | 'info' | 'warning' | 'error'}>({
    message: '',
    severity: 'info'
  });
  const [showNotification, setShowNotification] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [inspiredQuote, setInspiredQuote] = useState(getRandomQuote());
  
  // References
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));

  // Change quote every 30 seconds
  useEffect(() => {
    const quoteInterval = setInterval(() => {
      setInspiredQuote(getRandomQuote());
    }, 30000);
    
    return () => clearInterval(quoteInterval);
  }, []);

  // Show notification message
  const showMessage = (message: string, severity: 'success' | 'info' | 'warning' | 'error') => {
    setNotification({ message, severity });
    setShowNotification(true);
  };

  // Handle file upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) return;
    
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);
    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to upload file');
      }
      
      const data = await response.json();
      
      // Create audio URL for local playback
      const audioUrl = URL.createObjectURL(uploadedFile);
      setAudioInfo({ url: audioUrl, filename: data.filename });
      
      showMessage('File uploaded successfully', 'success');
    } catch (err: any) {
      setError(err.message || 'Error uploading file');
      showMessage(err.message || 'Error uploading file', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Use a local test file
  const useLocalFile = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/use_test_file`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to use test file');
      }
      
      const data = await response.json();
      
      // Set audio info with the server's test file
      setAudioInfo({ url: `${API_URL}/audio/${data.filename}`, filename: data.filename });
      
      showMessage('Test file loaded successfully', 'success');
    } catch (err: any) {
      setError(err.message || 'Error loading test file');
      showMessage(err.message || 'Error loading test file', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Analyze audio file
  const analyzeAudio = async () => {
    if (!audioInfo || !audioInfo.filename) {
      showMessage('Please upload an audio file first', 'warning');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setAnalysisStep(1);
    setAnalysisProgress(0);

    try {
      // Step 1: Initial audio analysis (tempo, key, time signature)
      showMessage('Step 1/5: Basic audio analysis...', 'info');
      setAnalysisStep(1);
      setAnalysisProgress(10);
      const basicResponse = await fetch(`${API_URL}/analyze/${encodeURIComponent(audioInfo.filename)}`);
      if (!basicResponse.ok) {
        const errorData = await basicResponse.json();
        throw new Error(errorData.error || 'Failed to perform basic analysis');
      }
      const basicResult = await basicResponse.json();
      setAnalysisProgress(20);
      
      // Step 2: Chord analysis
      showMessage('Step 2/5: Detecting chords and harmonies...', 'info');
      setAnalysisStep(2);
      setAnalysisProgress(30);
      const chordsResponse = await fetch(`${API_URL}/analyze/chords/${encodeURIComponent(audioInfo.filename)}`);
      if (!chordsResponse.ok) {
        const errorData = await chordsResponse.json();
        throw new Error(errorData.error || 'Failed to analyze chords');
      }
      const chordsResult = await chordsResponse.json();
      setAnalysisProgress(50);
      
      // Step 3: Key and scale analysis
      showMessage('Step 3/5: Determining key and scales...', 'info');
      setAnalysisStep(3);
      setAnalysisProgress(60);
      const keyResponse = await fetch(`${API_URL}/analyze/key/${encodeURIComponent(audioInfo.filename)}`);
      if (!keyResponse.ok) {
        const errorData = await keyResponse.json();
        throw new Error(errorData.error || 'Failed to analyze key');
      }
      const keyResult = await keyResponse.json();
      setAnalysisProgress(70);
      
      // Step 4: Note detection with advanced algorithms
      showMessage('Step 4/5: Detecting individual notes...', 'info');
      setAnalysisStep(4);
      setAnalysisProgress(80);
      const notesResponse = await fetch(`${API_URL}/analyze/notes/${encodeURIComponent(audioInfo.filename)}?use_advanced=true&use_aubio=true&use_ml=true`);
      if (!notesResponse.ok) {
        const errorData = await notesResponse.json();
        throw new Error(errorData.error || 'Failed to analyze notes');
      }
      const notesResult = await notesResponse.json();
      setNotesData(notesResult);
      setAnalysisProgress(90);
      
      // Step 5: Generate tablature with music theory integration
      showMessage('Step 5/5: Generating optimized tablature...', 'info');
      setAnalysisStep(5);
      setAnalysisProgress(95);
      const tablatureResponse = await fetch(
        `${API_URL}/analyze/structure/${encodeURIComponent(audioInfo.filename)}?key_hint=${keyResult.key || ''}&use_music21=true&optimize=true`
      );
      if (!tablatureResponse.ok) {
        const errorData = await tablatureResponse.json();
        throw new Error(errorData.error || 'Failed to generate tablature');
      }
      const tablatureResult = await tablatureResponse.json();
      setTablatureData(tablatureResult.tablature);
      
      // Step 6: Generate fretboard visualizations
      showMessage('Generating fretboard visualizations...', 'info');
      setAnalysisProgress(98);
      const fretboardResponse = await fetch(`${API_URL}/analyze/fretboard/${encodeURIComponent(audioInfo.filename)}`);
      if (fretboardResponse.ok) {
        const fretboardResult = await fretboardResponse.json();
        setNotesData(prev => ({...prev, currentNotes: fretboardResult.positions || []}));
      }
      setAnalysisProgress(100);

      showMessage('Analysis complete', 'success');
    } catch (err: any) {
      setError(err.message || 'Error analyzing audio');
      showMessage(err.message || 'Error analyzing audio', 'error');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle audio time updates
  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration);
    }
  };

  // Set up audio player when audio URL changes
  useEffect(() => {
    if (audioRef.current && audioInfo?.url) {
      audioRef.current.src = audioInfo.url;
      audioRef.current.load();
    }
  }, [audioInfo?.url]);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        {/* Modern App Bar */}
        <AppBar position="fixed" color="transparent" elevation={0} sx={{ 
          backdropFilter: 'blur(10px)',
          backgroundColor: 'rgba(18, 18, 18, 0.8)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <Toolbar>
            <Box 
              component="img" 
              src="/assets/logo.svg" 
              alt="OliTunes Logo"
              sx={{ height: 35, mr: 2 }}
            />
            
            <Typography 
              variant="h5" 
              className="shimmer" 
              sx={{ flexGrow: 1, fontWeight: 600 }}
            >
              OliTunes
            </Typography>
            
            {!isMobile && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Button 
                  startIcon={<HomeIcon />} 
                  color="inherit"
                  onClick={() => setCurrentTab(0)}
                  sx={{ opacity: currentTab === 0 ? 1 : 0.7 }}
                >
                  Home
                </Button>
                <Button 
                  startIcon={<LibraryMusicIcon />} 
                  color="inherit"
                  onClick={() => setCurrentTab(1)}
                  sx={{ opacity: currentTab === 1 ? 1 : 0.7 }}
                >
                  Analysis
                </Button>
                <Button 
                  startIcon={<MusicNoteIcon />} 
                  color="inherit"
                  onClick={() => setCurrentTab(2)}
                  sx={{ opacity: currentTab === 2 ? 1 : 0.7 }}
                >
                  Tablature
                </Button>
                <Button 
                  startIcon={<EqualizerIcon />} 
                  color="inherit"
                  onClick={() => setCurrentTab(3)}
                  sx={{ opacity: currentTab === 3 ? 1 : 0.7 }}
                >
                  Instruments
                </Button>
              </Box>
            )}
            
            <IconButton 
              color="inherit" 
              onClick={() => setShowSettings(true)}
              sx={{ ml: 2 }}
            >
              <SettingsIcon />
            </IconButton>
            
            {isMobile && (
              <IconButton 
                color="inherit" 
                onClick={() => setDrawerOpen(true)}
                edge="end"
              >
                <MenuIcon />
              </IconButton>
            )}
          </Toolbar>
        </AppBar>
        
        {/* Mobile Navigation Drawer */}
        <Drawer
          anchor="right"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
        >
          <Box sx={{ width: 250, pt: 2 }}>
            <List>
              <ListItem button onClick={() => { setCurrentTab(0); setDrawerOpen(false); }}>
                <ListItemIcon><HomeIcon color={currentTab === 0 ? "primary" : "inherit"} /></ListItemIcon>
                <ListItemText primary="Home" />
              </ListItem>
              <ListItem button onClick={() => { setCurrentTab(1); setDrawerOpen(false); }}>
                <ListItemIcon><LibraryMusicIcon color={currentTab === 1 ? "primary" : "inherit"} /></ListItemIcon>
                <ListItemText primary="Analysis" />
              </ListItem>
              <ListItem button onClick={() => { setCurrentTab(2); setDrawerOpen(false); }}>
                <ListItemIcon><MusicNoteIcon color={currentTab === 2 ? "primary" : "inherit"} /></ListItemIcon>
                <ListItemText primary="Tablature" />
              </ListItem>
              <ListItem button onClick={() => { setCurrentTab(3); setDrawerOpen(false); }}>
                <ListItemIcon><EqualizerIcon color={currentTab === 3 ? "primary" : "inherit"} /></ListItemIcon>
                <ListItemText primary="Instruments" />
              </ListItem>
            </List>
          </Box>
        </Drawer>
        
        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            pt: 8, // Space for AppBar
            minHeight: '100vh',
            backgroundImage: 'url(/guitar-background.svg)',
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundAttachment: 'fixed',
          }}
        >
          {/* Hero Section */}
          {!audioInfo && (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                p: 3,
                minHeight: '70vh',
                textAlign: 'center',
              }}
              className="fade-in"
            >
              <Typography variant="h1" component="h1" className="shimmer" gutterBottom>
                OliTunes
              </Typography>
              
              <Typography variant="h5" gutterBottom sx={{ maxWidth: 700, mb: 4 }} className="fade-in-delay-1">
                Music Analysis and Decomposition Platform
              </Typography>
              
              <Paper
                elevation={3}
                sx={{
                  p: 3,
                  mb: 4,
                  maxWidth: 500,
                  width: '100%',
                  backgroundColor: 'rgba(30, 30, 30, 0.7)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
                className="music-paper fade-in-delay-2"
              >
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    <MusicNoteIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Upload Your Music
                  </Typography>
                  <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                    Upload an audio file to analyze chord progressions, generate tablature, and more
                  </Typography>
                </Box>
                
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  startIcon={<CloudUploadIcon />}
                  sx={{ py: 1.5, mb: 2 }}
                  onClick={() => fileInputRef.current?.click()}
                  className="pulse"
                >
                  Select Audio File
                </Button>
                
                <input
                  type="file"
                  accept="audio/*"
                  hidden
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                />
                
                {file && (
                  <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
                    <Chip
                      icon={<AudioFileIcon />}
                      label={file.name}
                      color="secondary"
                      onDelete={() => setFile(null)}
                      sx={{ mr: 1, mb: 1 }}
                    />
                    <Button
                      variant="outlined"
                      onClick={analyzeAudio}
                      disabled={isAnalyzing}
                      sx={{ mb: 1 }}
                    >
                      {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                    </Button>
                  </Box>
                )}
              </Paper>
              
              <Box sx={{ maxWidth: 700, mb: 4 }} className="fade-in-delay-3">
                <Paper
                  elevation={2}
                  sx={{
                    p: 3,
                    backgroundColor: 'rgba(30, 30, 30, 0.6)',
                    backdropFilter: 'blur(5px)',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    borderLeft: '4px solid',
                    borderLeftColor: 'primary.main',
                  }}
                >
                  <Typography variant="h6" sx={{ fontStyle: 'italic', mb: 1 }}>
                    "{inspiredQuote.quote}"
                  </Typography>
                  <Typography variant="subtitle2" color="textSecondary" align="right">
                    — {inspiredQuote.author}
                  </Typography>
                </Paper>
              </Box>
              
              <Grid container spacing={3} sx={{ maxWidth: 1000 }} className="fade-in-delay-4">
                <Grid item xs={12} sm={6} md={3}>
                  <Card sx={{ height: '100%', backgroundColor: 'rgba(30, 30, 30, 0.7)' }}>
                    <CardContent>
                      <MusicNoteIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                      <Typography variant="h6" gutterBottom>
                        Note Detection
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Accurate pitch detection and note identification
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card sx={{ height: '100%', backgroundColor: 'rgba(30, 30, 30, 0.7)' }}>
                    <CardContent>
                      <LibraryMusicIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                      <Typography variant="h6" gutterBottom>
                        Chord Analysis
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Identify chords and progressions
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card sx={{ height: '100%', backgroundColor: 'rgba(30, 30, 30, 0.7)' }}>
                    <CardContent>
                      <MusicVideoIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                      <Typography variant="h6" gutterBottom>
                        Tablature Generation
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Generate guitar tablature from audio
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card sx={{ height: '100%', backgroundColor: 'rgba(30, 30, 30, 0.7)' }}>
                    <CardContent>
                      <EqualizerIcon color="primary" sx={{ fontSize: 40, mb: 1 }} />
                      <Typography variant="h6" gutterBottom>
                        Instrument Separation
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Extract individual instruments from mix
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          )}
          
          {/* Main Content Area */}
          {audioInfo && (
            <Grid container spacing={2} sx={{ p: 2, flexGrow: 1, height: 'calc(100vh - 80px)' }}>
              {/* Audio Controls Panel */}
              <Grid item xs={12} md={3} lg={2}>
                <Paper sx={{ 
                  p: 2, 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  backgroundColor: 'rgba(25,25,25,0.9)'
                }}>
                  <Typography variant="h6" sx={{ mb: 2 }}>Audio Controls</Typography>
                  
                  <MuiStack spacing={2} sx={{ mb: 3 }}>
                    <Button
                      fullWidth
                      variant="contained"
                      component="label"
                      startIcon={<CloudUploadIcon />}
                      disabled={isLoading}
                      sx={{ bgcolor: '#00c2cb', color: '#121212', '&:hover': { bgcolor: '#00a2a8' } }}
                    >
                      Upload Audio
                      <input
                        type="file"
                        hidden
                        accept="audio/*"
                        onChange={handleFileUpload}
                        ref={fileInputRef}
                      />
                    </Button>
                    
                    <Button
                      fullWidth
                      variant="outlined"
                      onClick={useLocalFile}
                      disabled={isLoading}
                      sx={{ borderColor: '#9c64a6', color: '#9c64a6' }}
                    >
                      Use Test File
                    </Button>
                    
                    <Button
                      fullWidth
                      variant="contained"
                      onClick={analyzeAudio}
                      disabled={isAnalyzing || !audioInfo}
                      sx={{ 
                        bgcolor: isAnalyzing ? 'rgba(25,25,25,0.5)' : '#9c64a6', 
                        color: '#ffffff',
                        '&:hover': { bgcolor: '#805584' },
                        '&.Mui-disabled': { bgcolor: 'rgba(156, 100, 166, 0.3)', color: 'rgba(255,255,255,0.3)' }
                      }}
                    >
                      {isAnalyzing ? (
                        <>
                          <CircularProgress size={20} sx={{ mr: 1, color: 'rgba(255,255,255,0.5)' }} />
                          Analyzing...
                        </>
                      ) : 'Analyze Audio'}
                    </Button>
                  </MuiStack>
                  
                  {audioInfo && (
                    <Box sx={{ 
                      display: 'flex', 
                      flexDirection: 'column', 
                      alignItems: 'center',
                      p: 2,
                      borderRadius: 1,
                      bgcolor: 'rgba(0,0,0,0.3)',
                      mt: 'auto'
                    }}>
                      <audio
                        ref={audioRef}
                        controls
                        style={{ width: '100%' }}
                        onTimeUpdate={handleTimeUpdate}
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                      />
                      
                      <Typography variant="caption" sx={{ mt: 1, color: 'rgba(255,255,255,0.7)' }}>
                        {Math.floor(currentTime / 60)}:{Math.floor(currentTime % 60).toString().padStart(2, '0')} / 
                        {Math.floor(duration / 60)}:{Math.floor(duration % 60).toString().padStart(2, '0')}
                      </Typography>
                    </Box>
                  )}
                </Paper>
              </Grid>
              
              {/* Main Visualization Area */}
              <Grid item xs={12} md={9} lg={10}>
                <Grid container spacing={2} sx={{ height: '100%' }}>
                  {/* Guitar Fretboard Visualization - Takes up top 60% of height */}
                  <Grid item xs={12} sx={{ height: '60%' }}>
                    <GuitarFretboard 
                      activeFrets={notesData?.currentNotes || []} 
                      highlightScale={notesData?.key || tablatureData?.key || null}
                      maxFrets={24}
                      height="100%"
                      showAllNotes={false}
                      darkMode={true}
                    />
                  </Grid>
                  
                  {/* Tablature Display - Takes up bottom 40% of height */}
                  <Grid item xs={12} sx={{ height: '40%' }}>
                    <TablatureDisplay 
                      tablature={tablatureData} 
                      currentTime={currentTime} 
                      notePositions={notesData?.notes || []}
                      darkMode={true}
                    />
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          )}
          
          {/* Notifications */}
          <Snackbar 
            open={showNotification} 
            autoHideDuration={6000} 
            onClose={() => setShowNotification(false)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          >
            <Alert 
              onClose={() => setShowNotification(false)} 
              severity={notification.severity} 
              sx={{ width: '100%' }}
            >
              {notification.message}
            </Alert>
          </Snackbar>
          
          {/* Loading Backdrop */}
          <Backdrop
            sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
            open={isLoading}
          >
            <CircularProgress color="inherit" />
          </Backdrop>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
