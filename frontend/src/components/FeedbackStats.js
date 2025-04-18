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
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { 
  Refresh, 
  InsertChart, 
  ThumbUp, 
  Star, 
  MusicNote, 
  EmojiEvents, 
  TrendingUp,
  TrendingDown,
  Equalizer
} from '@mui/icons-material';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell 
} from 'recharts';
import apiService from '../apiService';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];
const DIFFICULTY_COLORS = {
  'beginner': '#4caf50',
  'medium': '#ff9800',
  'advanced': '#f44336'
};

const FeedbackStats = () => {
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.getFeedbackStats();
      setStats(response.stats);
      setError(null);
    } catch (error) {
      console.error('Error fetching feedback stats:', error);
      setError('Failed to load feedback statistics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  // Prepare chart data
  const prepareRatingDistribution = () => {
    if (!stats || !stats.rating_distribution) return [];
    
    return Object.entries(stats.rating_distribution).map(([rating, count]) => ({
      name: `${rating} Star${rating !== '1' ? 's' : ''}`,
      value: count
    }));
  };

  const prepareDifficultyDistribution = () => {
    if (!stats || !stats.difficulty_distribution) return [];
    
    return Object.entries(stats.difficulty_distribution).map(([difficulty, count]) => ({
      name: difficulty.charAt(0).toUpperCase() + difficulty.slice(1),
      value: count,
      color: DIFFICULTY_COLORS[difficulty] || '#757575'
    }));
  };

  const prepareRatingBreakdown = () => {
    if (!stats || !stats.rating_breakdown) return [];
    
    return [
      {
        name: 'Playability',
        value: stats.rating_breakdown.playability
      },
      {
        name: 'Accuracy',
        value: stats.rating_breakdown.accuracy
      },
      {
        name: 'Musicality',
        value: stats.rating_breakdown.musicality
      }
    ];
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
          onClick={fetchStats}
          sx={{ mt: 2 }}
        >
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ mt: 3, mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">
          User Feedback Statistics
        </Typography>
        <Button 
          startIcon={<Refresh />} 
          variant="outlined" 
          onClick={fetchStats}
        >
          Refresh
        </Button>
      </Box>
      
      {stats && (
        <>
          {/* Summary Cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <ThumbUp color="primary" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography variant="h4" color="primary">
                    {stats.total_feedbacks}
                  </Typography>
                  <Typography variant="subtitle1">
                    Total Feedbacks
                  </Typography>
                </Box>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Star color="primary" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography variant="h4" color="primary">
                    {stats.average_rating.toFixed(1)}
                  </Typography>
                  <Typography variant="subtitle1">
                    Average Rating
                  </Typography>
                </Box>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <EmojiEvents color="primary" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography variant="h4" color="primary">
                    {stats.top_rated_model || 'N/A'}
                  </Typography>
                  <Typography variant="subtitle1">
                    Top Rated Model
                  </Typography>
                </Box>
              </Paper>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  {stats.rating_trend > 0 ? (
                    <TrendingUp color="success" sx={{ fontSize: 40, mb: 1 }} />
                  ) : (
                    <TrendingDown color="error" sx={{ fontSize: 40, mb: 1 }} />
                  )}
                  <Typography 
                    variant="h4" 
                    color={stats.rating_trend >= 0 ? 'success.main' : 'error.main'}
                  >
                    {stats.rating_trend > 0 ? '+' : ''}{stats.rating_trend.toFixed(1)}
                  </Typography>
                  <Typography variant="subtitle1">
                    Rating Trend
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
          
          {/* Charts */}
          <Grid container spacing={3}>
            {/* Rating Distribution */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Rating Distribution
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={prepareRatingDistribution()}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        nameKey="name"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {prepareRatingDistribution().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>
            
            {/* Difficulty Distribution */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Difficulty Distribution
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={prepareDifficultyDistribution()}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        nameKey="name"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      >
                        {prepareDifficultyDistribution().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <RechartsTooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>
            
            {/* Rating Breakdown */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Rating Breakdown
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={prepareRatingBreakdown()}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 5]} label={{ value: 'Average Rating', angle: -90, position: 'insideLeft' }} />
                      <RechartsTooltip />
                      <Legend />
                      <Bar dataKey="value" name="Average Rating" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>
          </Grid>
          
          {/* Recent Feedback */}
          {stats.recent_feedback && stats.recent_feedback.length > 0 && (
            <Paper sx={{ p: 2, mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Recent Feedback
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Date</TableCell>
                      <TableCell>Model</TableCell>
                      <TableCell>Rating</TableCell>
                      <TableCell>Difficulty</TableCell>
                      <TableCell>Comments</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {stats.recent_feedback.map((feedback, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          {new Date(feedback.timestamp).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={feedback.model_version || 'Unknown'} 
                            size="small" 
                            color="primary"
                          />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {feedback.rating}
                            <Star sx={{ fontSize: 16, ml: 0.5, color: '#FFB400' }} />
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={feedback.difficulty.charAt(0).toUpperCase() + feedback.difficulty.slice(1)} 
                            size="small"
                            sx={{ 
                              bgcolor: DIFFICULTY_COLORS[feedback.difficulty] || 'grey.500',
                              color: 'white'
                            }}
                          />
                        </TableCell>
                        <TableCell>
                          {feedback.comments 
                            ? feedback.comments.length > 50 
                              ? `${feedback.comments.substring(0, 50)}...` 
                              : feedback.comments
                            : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          )}
        </>
      )}
    </Box>
  );
};

export default FeedbackStats;
