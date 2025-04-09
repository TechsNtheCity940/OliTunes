import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  TextField, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  List, 
  ListItem, 
  ListItemText, 
  Divider, 
  CircularProgress,
  Chip,
  IconButton,
  Alert,
  Snackbar,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { 
  Search as SearchIcon, 
  Download as DownloadIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
  PlayArrow as ProcessIcon
} from '@mui/icons-material';
import apiService from '../../../src/apiService';

/**
 * Tab Manager component for searching, downloading, and managing guitar tabs
 * from Ultimate Guitar for model training
 */
const TabManager = () => {
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);

  // Download jobs state
  const [activeJobs, setActiveJobs] = useState([]);
  const [jobRefreshInterval, setJobRefreshInterval] = useState(null);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobDetails, setJobDetails] = useState(null);

  // Batch download state
  const [batchArtists, setBatchArtists] = useState('');
  const [songsPerArtist, setSongsPerArtist] = useState(10);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [batchJobId, setBatchJobId] = useState(null);

  // Notifications
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });

  // Load active jobs on component mount
  useEffect(() => {
    fetchActiveJobs();
    
    // Set up polling for job status updates
    const interval = setInterval(fetchActiveJobs, 5000);
    setJobRefreshInterval(interval);
    
    return () => {
      if (jobRefreshInterval) {
        clearInterval(jobRefreshInterval);
      }
    };
  }, []);
  
  // Update job details when selected job changes
  useEffect(() => {
    if (selectedJob) {
      fetchJobDetails(selectedJob);
    } else {
      setJobDetails(null);
    }
  }, [selectedJob]);

  // Handle tab search
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setIsSearching(true);
      setSearchError(null);
      const results = await apiService.searchTabs(searchQuery);
      setSearchResults(results.tabs || []);
      
      if ((results.tabs || []).length === 0) {
        setNotification({
          open: true,
          message: 'No tabs found for your search query',
          severity: 'info'
        });
      }
    } catch (error) {
      console.error('Error searching tabs:', error);
      setSearchError('Failed to search tabs. Please try again.');
      setNotification({
        open: true,
        message: 'Error searching tabs: ' + (error.message || 'Unknown error'),
        severity: 'error'
      });
    } finally {
      setIsSearching(false);
    }
  };

  // Handle tab download
  const handleDownload = async (tab) => {
    try {
      const response = await apiService.downloadTab(tab.id, tab.artist, tab.title);
      
      setNotification({
        open: true,
        message: `Started download for "${tab.title}" by ${tab.artist}`,
        severity: 'success'
      });
      
      // Refresh job list
      fetchActiveJobs();
    } catch (error) {
      console.error('Error downloading tab:', error);
      setNotification({
        open: true,
        message: 'Error downloading tab: ' + (error.message || 'Unknown error'),
        severity: 'error'
      });
    }
  };

  // Fetch active jobs from the server
  const fetchActiveJobs = async () => {
    try {
      const response = await apiService.listJobs();
      setActiveJobs(response.jobs || []);
      
      // Update selected job details if needed
      if (selectedJob && response.jobs) {
        const currentJob = response.jobs.find(job => job.id === selectedJob);
        if (currentJob) {
          setJobDetails(currentJob);
        }
      }
    } catch (error) {
      console.error('Error fetching active jobs:', error);
    }
  };

  // Fetch details for a specific job
  const fetchJobDetails = async (jobId) => {
    try {
      const response = await apiService.checkJobStatus(jobId);
      setJobDetails(response);
    } catch (error) {
      console.error(`Error fetching job details for ${jobId}:`, error);
      setJobDetails(null);
    }
  };

  // Handle job cancellation
  const handleCancelJob = async (jobId) => {
    try {
      await apiService.cancelJob(jobId);
      
      setNotification({
        open: true,
        message: 'Job cancelled successfully',
        severity: 'success'
      });
      
      // Refresh job list
      fetchActiveJobs();
      
      // Clear selected job if it was the cancelled one
      if (selectedJob === jobId) {
        setSelectedJob(null);
      }
    } catch (error) {
      console.error('Error cancelling job:', error);
      setNotification({
        open: true,
        message: 'Error cancelling job: ' + (error.message || 'Unknown error'),
        severity: 'error'
      });
    }
  };

  // Start batch download for multiple artists
  const handleBatchDownload = async () => {
    if (!batchArtists.trim()) {
      setNotification({
        open: true,
        message: 'Please enter at least one artist',
        severity: 'warning'
      });
      return;
    }
    
    try {
      setIsBatchProcessing(true);
      const artistList = batchArtists.split(',').map(artist => artist.trim());
      
      const response = await apiService.startBatchDownload(artistList, songsPerArtist);
      setBatchJobId(response.job_id);
      
      setNotification({
        open: true,
        message: `Started batch download for ${artistList.length} artists`,
        severity: 'success'
      });
      
      // Refresh job list
      fetchActiveJobs();
    } catch (error) {
      console.error('Error starting batch download:', error);
      setNotification({
        open: true,
        message: 'Error starting batch download: ' + (error.message || 'Unknown error'),
        severity: 'error'
      });
    } finally {
      setIsBatchProcessing(false);
    }
  };

  // Process existing tab files
  const handleProcessExistingTabs = async () => {
    try {
      const response = await apiService.processExistingTabs();
      
      setNotification({
        open: true,
        message: 'Started processing existing tab files',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error processing existing tabs:', error);
      setNotification({
        open: true,
        message: 'Error processing tabs: ' + (error.message || 'Unknown error'),
        severity: 'error'
      });
    }
  };

  // Close notification
  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  // Render job status chip
  const renderJobStatusChip = (status) => {
    switch (status) {
      case 'running':
        return <Chip color="primary" label="Running" />;
      case 'completed':
        return <Chip color="success" label="Completed" />;
      case 'failed':
        return <Chip color="error" label="Failed" />;
      case 'cancelled':
        return <Chip color="warning" label="Cancelled" />;
      default:
        return <Chip label={status} />;
    }
  };

  return (
    <Box sx={{ width: '100%', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Tab Manager
      </Typography>
      
      <Grid container spacing={3}>
        {/* Search and Results Section */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Search Tabs
            </Typography>
            
            <Box sx={{ display: 'flex', mb: 2 }}>
              <TextField
                fullWidth
                label="Search by Artist or Song"
                variant="outlined"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                sx={{ mr: 1 }}
              />
              <Button
                variant="contained"
                color="primary"
                onClick={handleSearch}
                disabled={isSearching || !searchQuery.trim()}
                startIcon={isSearching ? <CircularProgress size={20} /> : <SearchIcon />}
              >
                Search
              </Button>
            </Box>
            
            {searchError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {searchError}
              </Alert>
            )}
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                {searchResults.length > 0 ? `${searchResults.length} results found` : 'No results yet'}
              </Typography>
              
              <List>
                {searchResults.map((tab) => (
                  <React.Fragment key={tab.id}>
                    <ListItem
                      secondaryAction={
                        <IconButton 
                          edge="end" 
                          aria-label="download"
                          onClick={() => handleDownload(tab)}
                        >
                          <DownloadIcon />
                        </IconButton>
                      }
                    >
                      <ListItemText
                        primary={`${tab.title} - ${tab.artist}`}
                        secondary={`Rating: ${tab.rating}/5 | Type: ${tab.type}`}
                      />
                    </ListItem>
                    <Divider />
                  </React.Fragment>
                ))}
              </List>
            </Box>
          </Paper>
        </Grid>
        
        {/* Batch Download Section */}
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Batch Download
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <TextField
                fullWidth
                label="Artists (comma separated)"
                variant="outlined"
                value={batchArtists}
                onChange={(e) => setBatchArtists(e.target.value)}
                sx={{ mb: 2 }}
                placeholder="Led Zeppelin, Pink Floyd, Queen"
              />
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="songs-per-artist-label">Songs Per Artist</InputLabel>
                <Select
                  labelId="songs-per-artist-label"
                  value={songsPerArtist}
                  label="Songs Per Artist"
                  onChange={(e) => setSongsPerArtist(e.target.value)}
                >
                  <MenuItem value={5}>5</MenuItem>
                  <MenuItem value={10}>10</MenuItem>
                  <MenuItem value={20}>20</MenuItem>
                  <MenuItem value={50}>50</MenuItem>
                </Select>
              </FormControl>
              
              <Button
                fullWidth
                variant="contained"
                color="primary"
                onClick={handleBatchDownload}
                disabled={isBatchProcessing || !batchArtists.trim()}
                startIcon={isBatchProcessing ? <CircularProgress size={20} /> : <DownloadIcon />}
              >
                Start Batch Download
              </Button>
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="h6" gutterBottom>
              Tab Processing
            </Typography>
            
            <Button
              fullWidth
              variant="outlined"
              color="secondary"
              onClick={handleProcessExistingTabs}
              startIcon={<ProcessIcon />}
            >
              Process Existing Tabs
            </Button>
          </Paper>
        </Grid>
        
        {/* Active Jobs Section */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Active Jobs
              </Typography>
              
              <Button
                startIcon={<RefreshIcon />}
                onClick={fetchActiveJobs}
                size="small"
              >
                Refresh
              </Button>
            </Box>
            
            {activeJobs.length === 0 ? (
              <Alert severity="info">No active jobs</Alert>
            ) : (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>ID</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Progress</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {activeJobs.map((job) => (
                      <TableRow 
                        key={job.id}
                        hover
                        selected={selectedJob === job.id}
                        onClick={() => setSelectedJob(job.id)}
                      >
                        <TableCell>{job.id.substring(0, 8)}...</TableCell>
                        <TableCell>{job.type}</TableCell>
                        <TableCell>{renderJobStatusChip(job.status)}</TableCell>
                        <TableCell>
                          {job.status === 'running' && (
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <CircularProgress 
                                size={20} 
                                variant="determinate" 
                                value={job.progress * 100} 
                                sx={{ mr: 1 }} 
                              />
                              {Math.round(job.progress * 100)}%
                            </Box>
                          )}
                          {job.status !== 'running' && job.message}
                        </TableCell>
                        <TableCell>
                          {job.status === 'running' && (
                            <IconButton
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleCancelJob(job.id);
                              }}
                            >
                              <CancelIcon />
                            </IconButton>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
            
            {/* Job Details */}
            {jobDetails && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Job Details
                </Typography>
                
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="body2">
                      <strong>ID:</strong> {jobDetails.id}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Status:</strong> {jobDetails.status}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Created:</strong> {new Date(jobDetails.created_at).toLocaleString()}
                    </Typography>
                    {jobDetails.completed_at && (
                      <Typography variant="body2">
                        <strong>Completed:</strong> {new Date(jobDetails.completed_at).toLocaleString()}
                      </Typography>
                    )}
                    {jobDetails.results && (
                      <>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          <strong>Results:</strong>
                        </Typography>
                        <List dense>
                          {jobDetails.results.map((result, index) => (
                            <ListItem key={index}>
                              <ListItemText
                                primary={result.title}
                                secondary={`${result.artist} | Status: ${result.status}`}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </>
                    )}
                  </CardContent>
                </Card>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
      
      {/* Notifications */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity} 
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default TabManager;
