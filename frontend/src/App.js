import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [grid, setGrid] = useState([]); // 5x5 word grid
  const [status, setStatus] = useState(''); // Game status

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/start-game')
      .then(response => response.json())
      .then(data => setGrid(data.grid))
      .catch(error => console.error('Error fetching game grid:', error));
  }, []);

  const handleWordClick = (word) => {
    fetch('http://127.0.0.1:8000/api/guess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word }),
    })
      .then(response => response.json())
      .then(data => {
        setStatus(data.status);
        setGrid(data.grid);
      })
      .catch(error => console.error('Error making a guess:', error));
  };

  return (
    <div className="App">
      <h1>Codenames Duet AI</h1>
      <div className="grid">
        {grid.map((row, rowIndex) => (
          <div className="row" key={rowIndex}>
            {row.map((word, colIndex) => (
              <button 
                className="word-card" 
                key={colIndex} 
                onClick={() => handleWordClick(word)}>
                {word}
              </button>
            ))}
          </div>
        ))}
      </div>
      <p>{status}</p>
    </div>
  );
}

export default App;