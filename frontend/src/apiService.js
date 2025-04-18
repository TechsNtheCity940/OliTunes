import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

const apiService = {
  // Return the base URL for displaying in error messages
  getBaseUrl: () => API_BASE_URL,
  uploadAudio: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error uploading audio:', error);
      throw error;
    }
  },

  getAudioAnalysis: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting audio analysis:', error);
      throw error;
    }
  },

  getChords: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/chords/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting chord analysis:', error);
      throw error;
    }
  },

  getKey: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/key/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting key analysis:', error);
      throw error;
    }
  },

  getTimeSignature: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/time-signature/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting time signature:', error);
      throw error;
    }
  },

  getNotes: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/notes/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting note transcription:', error);
      throw error;
    }
  },

  getSeparatedInstruments: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/separate/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting instrument separation:', error);
      throw error;
    }
  },

  getTablature: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/tablature/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting tablature:', error);
      throw error;
    }
  },

  getLyrics: async (filename) => {
    try {
      // Replace spaces with underscores in the filename
      const sanitizedFilename = filename.replace(/\s+/g, '_');
      const response = await axios.get(`${API_BASE_URL}/analyze/lyrics/${encodeURIComponent(sanitizedFilename)}`);
      return response.data;
    } catch (error) {
      console.error('Error getting lyrics:', error);
      throw error;
    }
  },

  getStructureAnalysis: async (filename) => {
    try {
      // Replace spaces with underscores in the filename
      const sanitizedFilename = filename.replace(/\s+/g, '_');
      const response = await axios.get(`${API_BASE_URL}/analyze/structure/${encodeURIComponent(sanitizedFilename)}`);
      return response.data;
    } catch (error) {
      console.error('Error getting structure analysis:', error);
      throw error;
    }
  },
  
  // New enhanced tablature analysis methods
  getStructure: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/structure/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting song structure analysis:', error);
      throw error;
    }
  },
  
  getFretboardPositions: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/fretboard/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting fretboard positions:', error);
      throw error;
    }
  },
  
  getTextTablature: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/text_tablature/${encodeURIComponent(filename)}`);
      return response.data;
    } catch (error) {
      console.error('Error getting text tablature:', error);
      throw error;
    }
  },
  
  generateAdvancedTablature: async (filename) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate-advanced-tab/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error generating advanced tablature:', error);
      throw error;
    }
  },
  
  getTabJobStatus: async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/tab-job/${jobId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting tab job status:', error);
      throw error;
    }
  },
  
  getTabFile: async (jobId, filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/tab-job/${jobId}/file/${filename}`, {
        responseType: 'blob'
      });
      return response.data;
    } catch (error) {
      console.error('Error getting tab file:', error);
      throw error;
    }
  },

  playAudio: (filename) => {
    return `${API_BASE_URL}/play/${filename}`;
  },
  
  // New feedback and model performance methods
  submitFeedback: async (feedback) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/feedback/submit`, feedback);
      return response.data;
    } catch (error) {
      console.error('Error submitting feedback:', error);
      throw error;
    }
  },
  
  getFeedbackStats: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/feedback/stats`);
      return response.data;
    } catch (error) {
      console.error('Error getting feedback stats:', error);
      throw error;
    }
  },
  
  getModelPerformance: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model/performance`);
      return response.data;
    } catch (error) {
      console.error('Error getting model performance stats:', error);
      throw error;
    }
  },
  
  getModelCalibration: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model/calibration`);
      return response.data;
    } catch (error) {
      console.error('Error getting model calibration data:', error);
      throw error;
    }
  },
  
  // Tab Manager API functions
  searchTabs: async (query) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/tab_extractor/search?q=${encodeURIComponent(query)}`);
      return response.data;
    } catch (error) {
      console.error('Error searching tabs:', error);
      throw error;
    }
  },
  
  downloadTab: async (tabId, artist, title) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/tab_extractor/download`, {
        tab_id: tabId,
        artist,
        title
      });
      return response.data;
    } catch (error) {
      console.error('Error downloading tab:', error);
      throw error;
    }
  },
  
  checkJobStatus: async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/tab_extractor/job/${jobId}`);
      return response.data;
    } catch (error) {
      console.error('Error checking job status:', error);
      throw error;
    }
  },
  
  listJobs: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/tab_extractor/jobs`);
      return response.data;
    } catch (error) {
      console.error('Error listing jobs:', error);
      throw error;
    }
  },
  
  cancelJob: async (jobId) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/tab_extractor/cancel/${jobId}`);
      return response.data;
    } catch (error) {
      console.error('Error canceling job:', error);
      throw error;
    }
  },
  
  getBatchDownloadStatus: async (batchId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/tab_extractor/batch/${batchId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting batch download status:', error);
      throw error;
    }
  },
  
  startBatchDownload: async (artists, songsPerArtist) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/tab_extractor/extract`, {
        artists,
        limit: songsPerArtist
      });
      return response.data;
    } catch (error) {
      console.error('Error starting batch download:', error);
      throw error;
    }
  },
  
  processExistingTabs: async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/tab_extractor/process_existing_tabs`);
      return response.data;
    } catch (error) {
      console.error('Error processing existing tabs:', error);
      throw error;
    }
  },
  
  // Training data API functions
  trainModel: async (config) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/model/train`, config);
      return response.data;
    } catch (error) {
      console.error('Error training model:', error);
      throw error;
    }
  },
  
  getTrainingStatus: async (trainingId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model/training-status/${trainingId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting training status:', error);
      throw error;
    }
  },
  
  // Piano keyboard data
  getPianoNotes: async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/analyze/piano-notes/${filename}`);
      return response.data;
    } catch (error) {
      console.error('Error getting piano notes:', error);
      throw error;
    }
  },
  
  // Beat pattern functions
  saveBeatPattern: async (pattern) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/rhythm/save-pattern`, {
        pattern
      });
      return response.data;
    } catch (error) {
      console.error('Error saving beat pattern:', error);
      throw error;
    }
  },
  
  getBeatPatterns: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/rhythm/patterns`);
      return response.data;
    } catch (error) {
      console.error('Error getting beat patterns:', error);
      throw error;
    }
  }
};

export default apiService;
