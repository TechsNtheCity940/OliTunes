import React, { useState } from 'react';
import axios from 'axios';

const TabGenerator = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tabs, setTabs] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    const formData = new FormData();
    formData.append('audio', file);
    
    try {
      const response = await axios.post('/generate_tab', formData);
      setTabs(response.data);
    } catch (error) {
      console.error('Error generating tabs:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tab-generator">
      <form onSubmit={handleSubmit}>
        <input type="file" accept="audio/*" onChange={(e) => setFile(e.target.files[0])} />
        <button type="submit" disabled={!file || loading}>
          {loading ? 'Processing...' : 'Generate Tab'}
        </button>
      </form>
      
      {tabs && (
        <div className="tab-results">
          <pre>{tabs.tabs.guitar.text_tab}</pre>
        </div>
      )}
    </div>
  );
};

export default TabGenerator;
