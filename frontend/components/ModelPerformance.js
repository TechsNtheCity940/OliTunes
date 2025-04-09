import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  CircularProgress, 
  Divider, 
  Grid, 
  Paper, 
  Typography, 
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Button,
  Chip
} from '@mui/material';
import { 
  Timeline, 
  Memory, 
  Speed, 
  Cached, 
  Storage, 
  Assessment,
  BugReport,
  CheckCircle,
  Refresh
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import apiService from '../../../src/apiService';

const ModelPerformance = () => {
  const [performanceStats, setPerformanceStats] = useState(null);
  const [calibrationData, setCalibrationData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPerformanceData = async () => {
    setIsLoading(true);
    try {
      const performanceResponse = await apiService.getModelPerformance();
      setPerformanceStats(performanceResponse.stats);
      
      const calibrationResponse = await apiService.getModelCalibration();
      setCalibrationData(calibrationResponse.calibration);
      
      setError(null);
    } catch (error) {
      console.error('Error fetching model performance data:', error);
      setError('Failed to load model performance data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPerformanceData();
  }, []);

  // Format numbers for display
  const formatTime = (milliseconds) => {
    return milliseconds < 1000 
      ? `${milliseconds.toFixed(2)}ms` 
      : `${(milliseconds / 1000).toFixed(2)}s`;
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Prepare chart data
  const prepareTimeSeriesData = () => {
    if (!performanceStats || !performanceStats.time_series) return [];
    
    return performanceStats.time_series.map(item => ({
      timestamp: new Date(item.timestamp).toLocaleTimeString(),
      inferenceTime: item.inference_time,
      cacheHitRatio: item.cache_hit_ratio * 100
    }));
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">{error}</Typography>
        <Button 
          startIcon={<Refresh />} 
          variant="contained" 
          onClick={fetchPerformanceData}
          sx={{ mt: 2 }}
        >
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ mt: 3, mb: 3 }}>
      <Typography variant="h5" gutterBottom>
        Model Performance Metrics
      </Typography>
      
      {performanceStats && (
        <>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {/* Model Information */}
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Storage color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Model Information</Typography>
                </Box>
                <Divider sx={{ mb: 2 }} />
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Model Version:</Typography>
                  <Chip 
                    label={performanceStats.model_version} 
                    color="primary" 
                    size="small" 
                    sx={{ mt: 0.5 }}
                  />
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Quantized:</Typography>
                  <Typography>
                    {performanceStats.is_quantized ? 'Yes' : 'No'}
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Model Size:</Typography>
                  <Typography>
                    {(performanceStats.model_size_mb).toFixed(2)} MB
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Last Updated:</Typography>
                  <Typography>
                    {new Date(performanceStats.last_updated).toLocaleString()}
                  </Typography>
                </Box>
              </Paper>
            </Grid>
            
            {/* Cache Performance */}
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Cached color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Cache Performance</Typography>
                </Box>
                <Divider sx={{ mb: 2 }} />
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Cache Hit Ratio:</Typography>
                  <Typography variant="h4" color="success.main">
                    {formatPercentage(performanceStats.cache_hit_ratio)}
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Cache Size:</Typography>
                  <Typography>
                    {performanceStats.cache_size} entries
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Cache Hits:</Typography>
                  <Typography>
                    {performanceStats.cache_hits}
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Cache Misses:</Typography>
                  <Typography>
                    {performanceStats.cache_misses}
                  </Typography>
                </Box>
              </Paper>
            </Grid>
            
            {/* Timing Information */}
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Speed color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Timing Information</Typography>
                </Box>
                <Divider sx={{ mb: 2 }} />
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Avg. Inference Time:</Typography>
                  <Typography variant="h4" color="primary">
                    {formatTime(performanceStats.avg_inference_time)}
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Model Load Time:</Typography>
                  <Typography>
                    {formatTime(performanceStats.model_load_time)}
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">Batch Processing Time:</Typography>
                  <Typography>
                    {formatTime(performanceStats.avg_batch_time)}
                    <Typography variant="caption" sx={{ ml: 1 }}>
                      (batch size: {performanceStats.avg_batch_size || 'N/A'})
                    </Typography>
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
          
          {/* Performance Over Time Chart */}
          <Paper sx={{ p: 2, mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Timeline color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Performance Over Time</Typography>
            </Box>
            <Divider sx={{ mb: 2 }} />
            
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={prepareTimeSeriesData()}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis yAxisId="left" label={{ value: 'Inference Time (ms)', angle: -90, position: 'insideLeft' }} />
                  <YAxis yAxisId="right" orientation="right" label={{ value: 'Cache Hit Ratio (%)', angle: 90, position: 'insideRight' }} />
                  <RechartsTooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="inferenceTime" name="Inference Time (ms)" fill="#8884d8" />
                  <Bar yAxisId="right" dataKey="cacheHitRatio" name="Cache Hit Ratio (%)" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
          
          {/* Model Calibration Table */}
          {calibrationData && (
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Assessment color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Model Confidence Calibration</Typography>
              </Box>
              <Divider sx={{ mb: 2 }} />
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Confidence Bin</TableCell>
                      <TableCell align="right">Samples</TableCell>
                      <TableCell align="right">Accuracy</TableCell>
                      <TableCell align="right">Calibration Error</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {calibrationData.bins.map((bin, index) => (
                      <TableRow key={index}>
                        <TableCell>{`${bin.min_confidence * 100}% - ${bin.max_confidence * 100}%`}</TableCell>
                        <TableCell align="right">{bin.sample_count}</TableCell>
                        <TableCell align="right">{formatPercentage(bin.accuracy)}</TableCell>
                        <TableCell 
                          align="right"
                          sx={{ 
                            color: Math.abs(bin.calibration_error) > 0.1 ? 'error.main' : 'success.main' 
                          }}
                        >
                          {formatPercentage(bin.calibration_error)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2">
                  Overall Calibration Error: 
                  <Chip
                    label={formatPercentage(calibrationData.overall_error)}
                    color={calibrationData.overall_error < 0.1 ? "success" : "warning"}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                </Typography>
              </Box>
            </Paper>
          )}
        </>
      )}
    </Box>
  );
};

export default ModelPerformance;
