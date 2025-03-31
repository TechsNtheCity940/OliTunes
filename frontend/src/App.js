import React from 'react';

function App() {
    return (
        <div className="App">
            <h1>Audio to Sheet Music Converter</h1>
            <input type="file" accept="audio/*" />
            <button>Upload</button>
        </div>
    );
}

export default App;
