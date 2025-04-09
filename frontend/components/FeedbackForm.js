import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress, 
  Divider, 
  FormControl, 
  FormControlLabel, 
  FormLabel, 
  Grid, 
  MenuItem, 
  Paper, 
  Radio, 
  RadioGroup, 
  Rating, 
  Select, 
  Snackbar, 
  TextField, 
  Typography, 
  Alert 
} from '@mui/material';
import { ThumbUp, ThumbDown, Comment } from '@mui/icons-material';
import apiService from '../../../src/apiService';

const FeedbackForm = ({ tabId, positions, onFeedbackSubmitted }) => {
  const [rating, setRating] = useState(3);
  const [playability, setPlayability] = useState(3);
  const [accuracy, setAccuracy] = useState(3);
  const [musicality, setMusicality] = useState(3);
  const [comments, setComments] = useState('');
  const [difficulty, setDifficulty] = useState('medium');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const feedback = {
        rating,
        playability,
        accuracy,
        musicality,
        difficulty,
        comments,
        timestamp: new Date().toISOString()
      };

      const response = await apiService.submitFeedback(tabId, feedback, positions);
      
      setNotification({
        open: true,
        message: 'Thank you for your feedback!',
        severity: 'success'
      });

      // Reset form
      setRating(3);
      setPlayability(3);
      setAccuracy(3);
      setMusicality(3);
      setComments('');
      setDifficulty('medium');

      // Notify parent component
      if (onFeedbackSubmitted) {
        onFeedbackSubmitted(feedback);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setNotification({
        open: true,
        message: 'Error submitting feedback. Please try again.',
        severity: 'error'
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <Card variant="outlined" sx={{ mt: 3, mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Provide Feedback on This Tablature
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        <Box component="form" onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 3 }}>
                <Typography component="legend">Overall Rating</Typography>
                <Rating
                  name="rating"
                  value={rating}
                  onChange={(_, newValue) => setRating(newValue)}
                  precision={0.5}
                  size="large"
                />
              </Box>
              
              <Box sx={{ mb: 3 }}>
                <Typography component="legend">Playability</Typography>
                <Rating
                  name="playability"
                  value={playability}
                  onChange={(_, newValue) => setPlayability(newValue)}
                  precision={0.5}
                />
                <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                  How comfortable is it to play this tablature?
                </Typography>
              </Box>
              
              <Box sx={{ mb: 3 }}>
                <Typography component="legend">Accuracy</Typography>
                <Rating
                  name="accuracy"
                  value={accuracy}
                  onChange={(_, newValue) => setAccuracy(newValue)}
                  precision={0.5}
                />
                <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                  How accurately does this tablature represent the song?
                </Typography>
              </Box>
              
              <Box sx={{ mb: 3 }}>
                <Typography component="legend">Musicality</Typography>
                <Rating
                  name="musicality"
                  value={musicality}
                  onChange={(_, newValue) => setMusicality(newValue)}
                  precision={0.5}
                />
                <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                  How musical/expressive is this tablature?
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth sx={{ mb: 3 }}>
                <FormLabel id="difficulty-label">Difficulty Level</FormLabel>
                <RadioGroup
                  row
                  aria-labelledby="difficulty-label"
                  name="difficulty"
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                >
                  <FormControlLabel value="beginner" control={<Radio />} label="Beginner" />
                  <FormControlLabel value="medium" control={<Radio />} label="Intermediate" />
                  <FormControlLabel value="advanced" control={<Radio />} label="Advanced" />
                </RadioGroup>
              </FormControl>
              
              <TextField
                fullWidth
                label="Additional Comments"
                multiline
                rows={6}
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                variant="outlined"
                placeholder="Share your thoughts about this tablature..."
                sx={{ mb: 3 }}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={isSubmitting}
                startIcon={isSubmitting ? <CircularProgress size={20} /> : <Comment />}
                sx={{ mt: 2 }}
              >
                {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
              </Button>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
      
      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Card>
  );
};

export default FeedbackForm;
