import React, { useState, useRef, useEffect } from 'react';
import {
  Box, Button, Typography, Paper, Grid,
  Snackbar, Alert, Chip, IconButton,
  Drawer, List, ListItem, ListItemText, ListItemIcon, useTheme,
  CssBaseline, CircularProgress,
  Backdrop, Stack as MuiStack, Card, CardContent, 
  AppBar, Toolbar, useMediaQuery, LinearProgress
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  MusicNote as MusicNoteIcon,
  Settings as SettingsIcon,
  AudioFile as AudioFileIcon,
  Home as HomeIcon,
  LibraryMusic as LibraryMusicIcon,
  Equalizer as EqualizerIcon,
  MusicVideo as MusicVideoIcon,
  Menu as MenuIcon
} from '@mui/icons-material';
import './App.css';

// Import all components from a single entry point
import { 
  GuitarFretboard, 
  TablatureDisplay
} from './components';

// Import additional assets
import { getRandomQuote } from './assets/musicQuotes';
import './assets/animations.css';

// API service for backend communication
import apiService from './apiService';

function App() {
  // Define interfaces for our state types
  interface AudioInfo {
    url: string;
    filename: string;
  }
  
  interface NotesData {
    currentNotes?: any[];
    notes?: any[];
    key?: string;
  }
  
  interface TablatureData {
    key?: string;
    // Add other tablature properties as needed
  }
  
  type NotificationSeverity = 'success' | 'info' | 'warning' | 'error';

  interface AnalysisProgress {
    step: number;
    totalSteps: number;
    currentOperation: string;
    percentage: number;
  }
  
  // State for file handling and audio
  const [file, setFile] = useState<File | null>(null);
  const [audioInfo, setAudioInfo] = useState<AudioInfo | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [currentTab, setCurrentTab] = useState<number>(0);
  
  // Analysis states
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>({
    step: 0,
    totalSteps: 7,
    currentOperation: '',
    percentage: 0
  });
  
  // UI states
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);
  const [notification, setNotification] = useState<{message: string, severity: NotificationSeverity}>({ 
    message: '', 
    severity: 'info' 
  });
  const [showNotification, setShowNotification] = useState<boolean>(false);
  const [inspiredQuote, setInspiredQuote] = useState(getRandomQuote());
  
  // Analysis results
  const [notesData, setNotesData] = useState<NotesData | null>(null);
  const [tablatureData, setTablatureData] = useState<TablatureData | null>(null);
  
  // References
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Show notification message
  const showMessage = (message: string, severity: NotificationSeverity) => {
    setNotification({ message, severity });
    setShowNotification(true);
  };

  // Update audio info and source
  const handleAudioSourceUpdate = (audioSrc: string, filename: string) => {
    setAudioInfo({ 
      url: audioSrc, 
      filename 
    });
  };

  // Handle file upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) {
      return;
    }
    
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);
    setIsLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      
      const response = await fetch(`${apiService}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error uploading file');
      }
      
      const data = await response.json();
      
      // Set new audio info
      handleAudioSourceUpdate(`${apiService}/audio/${data.filename}`, data.filename);
      
      showMessage('File uploaded successfully', 'success');
    } catch (err: any) {
      showMessage(err.message || 'Error uploading file', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Use a local test file
  const useLocalFile = async () => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${apiService}/use_test_file`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error loading test file');
      }
      const data = await response.json();
      
      // Set audio info with the server's test file
      handleAudioSourceUpdate(`${apiService}/audio/${data.filename}`, data.filename);
      
      showMessage('Test file loaded successfully', 'success');
    } catch (err: any) {
      showMessage(err.message || 'Error loading test file', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle audio analysis
  const analyzeAudio = async () => {
    if (!audioInfo) {
      showMessage('No audio file loaded', 'error');
      return;
    }

    setIsAnalyzing(true);
    setNotesData(null);
    setTablatureData(null);
    
    // Reset and initialize analysis progress
    setAnalysisProgress({
      step: 1,
      totalSteps: 7, 
      currentOperation: 'Starting comprehensive analysis...',
      percentage: 5
    });

    try {
      // Use the comprehensive analysis endpoint
      showMessage('Starting comprehensive audio analysis...', 'info');
      
      // Update progress for step 1
      setAnalysisProgress({
        step: 1,
        totalSteps: 7,
        currentOperation: 'Analyzing audio fundamentals...',
        percentage: 15
      });
      
      const analysisResponse = await fetch(
        `${apiService}/analyze/comprehensive/${encodeURIComponent(audioInfo.filename)}`
      );
      
      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json();
        throw new Error(errorData.error || 'Error during comprehensive analysis');
      }
      
      // Since we don't have true streaming response from our backend endpoint yet,
      // we'll simulate progress updates as we process the response
      setAnalysisProgress({
        step: 2,
        totalSteps: 7,
        currentOperation: 'Analyzing tempo and time signature...',
        percentage: 25
      });
      
      setTimeout(() => {
        setAnalysisProgress({
          step: 3,
          totalSteps: 7,
          currentOperation: 'Detecting key and scale...',
          percentage: 40
        });
      }, 1000);
      
      setTimeout(() => {
        setAnalysisProgress({
          step: 4,
          totalSteps: 7, 
          currentOperation: 'Analyzing chord progressions...',
          percentage: 55
        });
      }, 2000);
      
      setTimeout(() => {
        setAnalysisProgress({
          step: 5,
          totalSteps: 7,
          currentOperation: 'Detecting guitar notes with frequency matching...',
          percentage: 70
        });
      }, 3000);
      
      setTimeout(() => {
        setAnalysisProgress({
          step: 6,
          totalSteps: 7,
          currentOperation: 'Generating optimized tablature...',
          percentage: 85
        });
      }, 4000);
      
      const analysisResult = await analysisResponse.json();
      
      // Final progress update
      setAnalysisProgress({
        step: 7,
        totalSteps: 7,
        currentOperation: 'Finalizing analysis...',
        percentage: 95
      });
      
      // Process the comprehensive results
      if (analysisResult.notes) {
        setNotesData(analysisResult.notes);
      }
      
      if (analysisResult.tablature) {
        setTablatureData(analysisResult.tablature);
      }
      
      // If fretboard positions are available, update the current notes
      if (analysisResult.fretboard && analysisResult.fretboard.positions) {
        setNotesData(prev => prev ? 
          {...prev, currentNotes: analysisResult.fretboard.positions} : 
          {currentNotes: analysisResult.fretboard.positions}
        );
      }
      
      // Complete progress
      setAnalysisProgress({
        step: 7,
        totalSteps: 7,
        currentOperation: 'Analysis complete',
        percentage: 100
      });
      
      showMessage('Analysis complete', 'success');
    } catch (err: any) {
      showMessage(err.message || 'Error analyzing audio', 'error');
      // Reset progress on error
      setAnalysisProgress({
        step: 0,
        totalSteps: 7,
        currentOperation: 'Analysis failed',
        percentage: 0
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle audio time update
  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration);
    }
  };

  // Load audio element when URL changes
  useEffect(() => {
    if (audioInfo && audioRef.current) {
      audioRef.current.src = audioInfo.url;
      audioRef.current.load();
    }
  }, [audioInfo]);

  // Change quote every 30 seconds
  useEffect(() => {
    const quoteInterval = setInterval(() => {
      setInspiredQuote(getRandomQuote());
    }, 30000);
    
    return () => clearInterval(quoteInterval);
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <CssBaseline />
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
            onClick={() => {}}
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
                  
                  {/* Analysis Progress Indicator */}
                  {isAnalyzing && (
                    <Box sx={{ width: '100%', mt: 2 }}>
                      <Typography variant="caption" color="textSecondary">
                        {analysisProgress.currentOperation} ({analysisProgress.step}/{analysisProgress.totalSteps})
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={analysisProgress.percentage} 
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  )}
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
                      onPlay={() => {}}
                      onPause={() => {}}
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
  );
}

export default App;
