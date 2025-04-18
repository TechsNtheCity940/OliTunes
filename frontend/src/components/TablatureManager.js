import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress, 
  Container, 
  Divider, 
  Grid, 
  LinearProgress, 
  List, 
  ListItem, 
  ListItemText, 
  Paper, 
  Tab, 
  Tabs, 
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Snackbar,
  Alert,
  Chip,
  ToggleButton,
  ToggleButtonGroup
} from '@mui/material';
import {
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Save as SaveIcon,
  Delete as DeleteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon
} from '@mui/icons-material';

// TabViewer component for displaying tablature
function TabViewer({ tabContent }) {
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        mt: 2, 
        fontFamily: 'monospace', 
        fontSize: '0.9rem',
        whiteSpace: 'pre-wrap',
        overflow: 'auto',
        maxHeight: '500px',
        backgroundColor: '#f8f9fa'
      }}
    >
      {tabContent || 'No tablature content to display'}
    </Paper>
  );
}

// Training progress visualization
function TrainingProgress({ progress, metrics }) {
  return (
    <Card sx={{ mt: 2, mb: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Training Progress
        </Typography>
        <Box sx={{ width: '100%', mb: 2 }}>
          <LinearProgress 
            variant="determinate" 
            value={progress * 100} 
            sx={{ height: 10, borderRadius: 5 }}
          />
          <Typography variant="body2" sx={{ mt: 1 }}>
            {Math.round(progress * 100)}% Complete
          </Typography>
        </Box>
        
        {metrics && (
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2">Accuracy: {metrics.accuracy.toFixed(2)}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">Loss: {metrics.loss.toFixed(4)}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">Epoch: {metrics.epoch}/{metrics.totalEpochs}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">Learning Rate: {metrics.learningRate.toExponential(2)}</Typography>
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  );
}

// Main TablatureManager component
function TablatureManager() {
  const [activeTab, setActiveTab] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [tabCollection, setTabCollection] = useState([]);
  const [selectedTab, setSelectedTab] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [artistFilter, setArtistFilter] = useState('');
  const [styleFilter, setStyleFilter] = useState('');
  const [downloadOptions, setDownloadOptions] = useState({
    artist: '',
    limit: 10,
    includeRated: true,
    minRating: 3
  });
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  const [viewMode, setViewMode] = useState('compact');

  // Simulate tab collection fetch
  useEffect(() => {
    // This would be replaced with an actual API call in production
    const mockTabCollection = [
      {
        id: 'tab1',
        title: 'Stairway to Heaven',
        artist: 'Led Zeppelin',
        style: 'rock',
        dateAdded: '2023-01-15',
        rating: 4.8,
        content: `
e|-------5-7-----7-|-8-----8-2-----2-|-0---------0-----|-----------------|
B|-----5-----5-----|---5-------3-----|---1---1-----1---|-0-1-1-----------|
G|---5---------5---|-----5-------2---|-----2---------2-|-0-2-2-----------|
D|-7-------6-------|-5-------4-------|-3---------------|-0-2-2-----------|
A|-----------------|-----------------|-----------------|-0-0-0-----------|
E|-----------------|-----------------|-----------------|-----------------|
        `
      },
      {
        id: 'tab2',
        title: 'Blackbird',
        artist: 'The Beatles',
        style: 'folk',
        dateAdded: '2023-02-10',
        rating: 4.5,
        content: `
e|---0-----0-7-|-0-----0-5-|-0-----0-3-|-0-----0-2-|
B|-----0--10---|-----0--8--|-----0--5--|-----0--3--|
G|-------9-----|-------7---|-------5---|-------2---|
D|-------------|-----------|-----------|-----------|
A|-------------|-----------|-----------|-----------|
E|-------------|-----------|-----------|-----------|
        `
      },
      {
        id: 'tab3',
        title: 'Hotel California',
        artist: 'Eagles',
        style: 'rock',
        dateAdded: '2023-03-22',
        rating: 4.2,
        content: `
e|--------7---7---|-8---8---7---7---|-----7---7---|-8---8---7---7---|
B|-----10-----10--|---10------10----|--10-----10--|---10------10----|
G|---9---------9--|------9-------9--|--9---------9|------9-------9--|
D|-----------------|-----------------|--------------|--------------10-|
A|-----------------|-----------------|--------------|-----------------|
E|-----------------|-----------------|--------------|-----------------|
        `
      },
      {
        id: 'tab4',
        title: 'Neon',
        artist: 'John Mayer',
        style: 'pop',
        dateAdded: '2023-04-05',
        rating: 4.9,
        content: `
e|-----------------|-----------------|-----------------|-----------------|
B|-----------------|-----------------|-----------------|-----------------|
G|-----------------|-----------------|-----------------|-----------------|
D|----7--7--7--7---|----7--7--7--7---|----9--9--9--9---|----7--7--7--7---|
A|--9-----9-----9--|--9-----9-----9--|--11----11----11-|--9-----9-----9--|
E|-7-----5-----3---|-2-----3-----5---|-7-----5-----7---|-5-----7-----9---|
        `
      },
      {
        id: 'tab5',
        title: 'Purple Haze',
        artist: 'Jimi Hendrix',
        style: 'rock',
        dateAdded: '2023-05-18',
        rating: 4.6,
        content: `
e|-----------------|-----------------|-----------------|-----------------|
B|-----------------|-----------------|-----------------|-----------------|
G|-----------------|-----------------|-----------------|-----------------|
D|-----------------|-----------------|-----------------|-----------------|
A|-7--7-7--7--7-7--|-7--7-7--7--7-10-|-7--7-7--7--7-7--|-7--7-7--7--7-6--|
E|-5--5-5--5--5-5--|-5--5-5--5--5-8--|-5--5-5--5--5-5--|-5--5-5--5--5-4--|
        `
      }
    ];
    
    setTabCollection(mockTabCollection);
  }, []);

  // Filter tabs based on search and filters
  const filteredTabs = tabCollection.filter(tab => {
    const matchesSearch = tab.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          tab.artist.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesArtist = artistFilter ? tab.artist === artistFilter : true;
    const matchesStyle = styleFilter ? tab.style === styleFilter : true;
    
    return matchesSearch && matchesArtist && matchesStyle;
  });

  // Get unique artists and styles for filters
  const uniqueArtists = [...new Set(tabCollection.map(tab => tab.artist))];
  const uniqueStyles = [...new Set(tabCollection.map(tab => tab.style))];

  // Tab change handler
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Select a tab for viewing
  const handleSelectTab = (tab) => {
    setSelectedTab(tab);
  };

  // Handle search input
  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
  };

  // Handle artist filter change
  const handleArtistFilterChange = (e) => {
    setArtistFilter(e.target.value);
  };

  // Handle style filter change
  const handleStyleFilterChange = (e) => {
    setStyleFilter(e.target.value);
  };

  // Handle download options change
  const handleDownloadOptionsChange = (e) => {
    const { name, value } = e.target;
    setDownloadOptions({
      ...downloadOptions,
      [name]: value
    });
  };

  // Toggle include rated option
  const handleToggleIncludeRated = () => {
    setDownloadOptions({
      ...downloadOptions,
      includeRated: !downloadOptions.includeRated
    });
  };

  // Simulate tab download
  const handleDownloadTabs = () => {
    setIsLoading(true);
    setDownloadProgress(0);
    
    // Mock download progress
    const interval = setInterval(() => {
      setDownloadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsLoading(false);
          setNotification({
            open: true,
            message: 'Tabs downloaded successfully!',
            severity: 'success'
          });
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  };

  // Simulate model training
  const handleTrainModel = () => {
    setIsLoading(true);
    setTrainingProgress(0);
    
    // Mock training progress
    let epoch = 0;
    const totalEpochs = 10;
    
    const interval = setInterval(() => {
      epoch++;
      const progress = epoch / totalEpochs;
      
      setTrainingProgress(progress);
      setTrainingMetrics({
        accuracy: 0.5 + (0.4 * progress),
        loss: 0.5 - (0.4 * progress),
        epoch,
        totalEpochs,
        learningRate: 0.001 * (1 - progress * 0.9)
      });
      
      if (epoch >= totalEpochs) {
        clearInterval(interval);
        setIsLoading(false);
        setNotification({
          open: true,
          message: 'Model training completed!',
          severity: 'success'
        });
      }
    }, 500);
  };

  // Handle notification close
  const handleCloseNotification = () => {
    setNotification({
      ...notification,
      open: false
    });
  };

  // Toggle view mode
  const handleViewModeChange = (event, newMode) => {
    if (newMode !== null) {
      setViewMode(newMode);
    }
  };

  // Rate a tab
  const handleRateTab = (tabId, newRating) => {
    setTabCollection(tabs => 
      tabs.map(tab => 
        tab.id === tabId ? {...tab, rating: newRating} : tab
      )
    );
    
    setNotification({
      open: true,
      message: 'Tab rating updated',
      severity: 'info'
    });
  };

  // Delete a tab
  const handleDeleteTab = (tabId) => {
    setTabCollection(tabs => tabs.filter(tab => tab.id !== tabId));
    
    if (selectedTab && selectedTab.id === tabId) {
      setSelectedTab(null);
    }
    
    setNotification({
      open: true,
      message: 'Tab deleted',
      severity: 'info'
    });
  };

  return (
    <Container>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mt: 3 }}>
        Tablature Manager
      </Typography>
      
      <Paper sx={{ width: '100%', mb: 2 }}>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
        >
          <Tab label="Browse Tabs" />
          <Tab label="Download" />
          <Tab label="Training" />
        </Tabs>
      </Paper>
      
      {/* Tab Panel 1: Browse Tabs */}
      {activeTab === 0 && (
        <Box>
          <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
            <TextField
              label="Search"
              variant="outlined"
              size="small"
              value={searchQuery}
              onChange={handleSearchChange}
              sx={{ flexGrow: 1, minWidth: '200px' }}
            />
            
            <FormControl size="small" sx={{ minWidth: '150px' }}>
              <InputLabel>Artist</InputLabel>
              <Select
                value={artistFilter}
                onChange={handleArtistFilterChange}
                label="Artist"
              >
                <MenuItem value="">
                  <em>All</em>
                </MenuItem>
                {uniqueArtists.map((artist, index) => (
                  <MenuItem key={index} value={artist}>{artist}</MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <FormControl size="small" sx={{ minWidth: '150px' }}>
              <InputLabel>Style</InputLabel>
              <Select
                value={styleFilter}
                onChange={handleStyleFilterChange}
                label="Style"
              >
                <MenuItem value="">
                  <em>All</em>
                </MenuItem>
                {uniqueStyles.map((style, index) => (
                  <MenuItem key={index} value={style}>{style}</MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={handleViewModeChange}
              size="small"
            >
              <ToggleButton value="compact">
                <VisibilityOffIcon fontSize="small" />
              </ToggleButton>
              <ToggleButton value="expanded">
                <VisibilityIcon fontSize="small" />
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
          
          {filteredTabs.length === 0 ? (
            <Paper sx={{ p: 3, textAlign: 'center' }}>
              <Typography>No tabs found matching your criteria</Typography>
            </Paper>
          ) : (
            <Grid container spacing={2}>
              {filteredTabs.map((tab) => (
                <Grid item xs={12} key={tab.id}>
                  <Card sx={{ display: 'flex', flexDirection: 'column' }}>
                    <CardContent sx={{ pb: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box>
                          <Typography variant="h6" component="div">
                            {tab.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {tab.artist}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                            <Chip size="small" label={tab.style} />
                            <Chip 
                              size="small" 
                              label={`Added: ${tab.dateAdded}`}
                              variant="outlined"
                            />
                          </Box>
                        </Box>
                        
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {/* Star rating */}
                          {[1, 2, 3, 4, 5].map((rating) => (
                            <IconButton 
                              key={rating} 
                              size="small"
                              onClick={() => handleRateTab(tab.id, rating)}
                            >
                              {rating <= tab.rating ? (
                                <StarIcon fontSize="small" color="primary" />
                              ) : (
                                <StarBorderIcon fontSize="small" />
                              )}
                            </IconButton>
                          ))}
                          
                          <IconButton 
                            size="small" 
                            color="error"
                            onClick={() => handleDeleteTab(tab.id)}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      </Box>
                      
                      {/* Display tab content in expanded view */}
                      {viewMode === 'expanded' && (
                        <TabViewer tabContent={tab.content} />
                      )}
                    </CardContent>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
                      <Button 
                        size="small" 
                        startIcon={<VisibilityIcon />}
                        onClick={() => handleSelectTab(tab)}
                      >
                        View
                      </Button>
                    </Box>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
          
          {/* Selected tab details */}
          {selectedTab && (
            <Card sx={{ mt: 3, mb: 3 }}>
              <CardContent>
                <Typography variant="h5" component="div" gutterBottom>
                  {selectedTab.title}
                </Typography>
                <Typography variant="subtitle1" color="text.secondary">
                  Artist: {selectedTab.artist}
                </Typography>
                <Typography variant="body2" mb={2}>
                  Style: {selectedTab.style} | Rating: {selectedTab.rating}/5
                </Typography>
                
                <Divider sx={{ mb: 2 }} />
                
                <Typography variant="h6" gutterBottom>
                  Tablature
                </Typography>
                
                <TabViewer tabContent={selectedTab.content} />
              </CardContent>
            </Card>
          )}
        </Box>
      )}
      
      {/* Tab Panel 2: Download */}
      {activeTab === 1 && (
        <Box>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Download Tabs
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Artist Name"
                    name="artist"
                    value={downloadOptions.artist}
                    onChange={handleDownloadOptionsChange}
                    margin="normal"
                    variant="outlined"
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Tab Limit"
                    name="limit"
                    value={downloadOptions.limit}
                    onChange={handleDownloadOptionsChange}
                    margin="normal"
                    variant="outlined"
                    inputProps={{ min: 1, max: 100 }}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                    <Typography variant="body2" sx={{ mr: 2 }}>
                      Include rated tabs only:
                    </Typography>
                    <ToggleButton
                      value="includeRated"
                      selected={downloadOptions.includeRated}
                      onChange={handleToggleIncludeRated}
                      size="small"
                    >
                      {downloadOptions.includeRated ? 'Yes' : 'No'}
                    </ToggleButton>
                  </Box>
                </Grid>
                
                {downloadOptions.includeRated && (
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Minimum Rating"
                      name="minRating"
                      value={downloadOptions.minRating}
                      onChange={handleDownloadOptionsChange}
                      margin="normal"
                      variant="outlined"
                      inputProps={{ min: 1, max: 5, step: 0.1 }}
                    />
                  </Grid>
                )}
                
                <Grid item xs={12}>
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<DownloadIcon />}
                      onClick={handleDownloadTabs}
                      disabled={isLoading || !downloadOptions.artist}
                      fullWidth
                    >
                      Download Tabs
                    </Button>
                  </Box>
                </Grid>
              </Grid>
              
              {isLoading && activeTab === 1 && (
                <Box sx={{ width: '100%', mt: 3 }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={downloadProgress} 
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                  <Typography variant="body2" align="center" sx={{ mt: 1 }}>
                    Downloading: {downloadProgress}%
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
          
          <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
            Recent Downloads
          </Typography>
          
          <List component={Paper}>
            {[
              { name: 'Led Zeppelin Collection', count: 24, date: '2023-05-15' },
              { name: 'Pink Floyd Collection', count: 18, date: '2023-05-10' },
              { name: 'The Beatles Collection', count: 32, date: '2023-05-01' }
            ].map((download, index) => (
              <React.Fragment key={index}>
                <ListItem>
                  <ListItemText
                    primary={download.name}
                    secondary={`${download.count} tabs • Downloaded on ${download.date}`}
                  />
                  <Button startIcon={<RefreshIcon />} size="small">
                    Refresh
                  </Button>
                </ListItem>
                {index < 2 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Box>
      )}
      
      {/* Tab Panel 3: Training */}
      {activeTab === 2 && (
        <Box>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Train Tablature Model
              </Typography>
              
              <Typography variant="body2" paragraph>
                Train the machine learning model to improve tab generation accuracy.
                This process will use your downloaded tabs as training data.
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Training Dataset</InputLabel>
                    <Select
                      value="all_tabs"
                      label="Training Dataset"
                    >
                      <MenuItem value="all_tabs">All Downloaded Tabs</MenuItem>
                      <MenuItem value="high_rated">High-Rated Tabs Only</MenuItem>
                      <MenuItem value="rock">Rock Tabs Only</MenuItem>
                      <MenuItem value="jazz">Jazz Tabs Only</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Training Mode</InputLabel>
                    <Select
                      value="full"
                      label="Training Mode"
                    >
                      <MenuItem value="full">Full Training</MenuItem>
                      <MenuItem value="fine_tune">Fine-Tuning</MenuItem>
                      <MenuItem value="transfer">Transfer Learning</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Model Architecture</InputLabel>
                    <Select
                      value="transformer"
                      label="Model Architecture"
                    >
                      <MenuItem value="transformer">Transformer (Best Performance)</MenuItem>
                      <MenuItem value="lstm">LSTM (Faster Training)</MenuItem>
                      <MenuItem value="hybrid">Hybrid (Balanced)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={handleTrainModel}
                      disabled={isLoading}
                      fullWidth
                    >
                      Start Training
                    </Button>
                  </Box>
                </Grid>
              </Grid>
              
              {isLoading && activeTab === 2 && (
                <Box sx={{ width: '100%', mt: 3, display: 'flex', alignItems: 'center' }}>
                  <CircularProgress size={24} sx={{ mr: 2 }} />
                  <Typography variant="body2">
                    Training in progress...
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
          
          {trainingProgress > 0 && (
            <TrainingProgress 
              progress={trainingProgress} 
              metrics={trainingMetrics} 
            />
          )}
          
          <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
            Training History
          </Typography>
          
          <List component={Paper}>
            {[
              { 
                name: 'Rock Style Fine-tuning', 
                date: '2023-05-10', 
                accuracy: '87.5%',
                dataset: 'Rock Tabs (146 samples)'
              },
              { 
                name: 'Full Model Training', 
                date: '2023-05-01', 
                accuracy: '82.3%',
                dataset: 'All Tabs (324 samples)'
              },
              { 
                name: 'Jazz Chord Training', 
                date: '2023-04-22', 
                accuracy: '79.8%',
                dataset: 'Jazz Tabs (68 samples)'
              }
            ].map((training, index) => (
              <React.Fragment key={index}>
                <ListItem>
                  <ListItemText
                    primary={training.name}
                    secondary={
                      <>
                        <Typography variant="body2" component="span">
                          {training.date} • Accuracy: {training.accuracy}
                        </Typography>
                        <br />
                        <Typography variant="body2" component="span" color="text.secondary">
                          Dataset: {training.dataset}
                        </Typography>
                      </>
                    }
                  />
                  <Button startIcon={<SaveIcon />} size="small" variant="outlined">
                    Export
                  </Button>
                </ListItem>
                {index < 2 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Box>
      )}
      
      {/* Notification */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default TablatureManager;
