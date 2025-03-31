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
      // Replace spaces with underscores in the filename
      const sanitizedFilename = filename.replace(/\s+/g, '_');
      const response = await axios.get(`${API_BASE_URL}/analyze/structure/${encodeURIComponent(sanitizedFilename)}`);
      return response.data;
    } catch (error) {
      console.error('Error getting enhanced tablature structure:', error);
      throw error;
    }
  },
  
  getFretboardPositions: async (filename) => {
    try {
      // Replace spaces with underscores in the filename
      const sanitizedFilename = filename.replace(/\s+/g, '_');
      const response = await axios.get(`${API_BASE_URL}/analyze/fretboard/${encodeURIComponent(sanitizedFilename)}`);
      return response.data;
    } catch (error) {
      console.error('Error getting fretboard positions:', error);
      throw error;
    }
  },

  playAudio: (filename) => {
    return `${API_BASE_URL}/play/${filename}`;
  }
};

export default apiService;
