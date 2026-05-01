import React, { useState } from 'react';
import './index.css';

function App() {
  const [viewMode, setViewMode] = useState('generate'); // 'generate' or 'history'
  const [username, setUsername] = useState('');
  
  // Form State
  const [ingredients, setIngredients] = useState('rice, egg, potato, tomato, onion');
  const [budget, setBudget] = useState(150);
  const [servings, setServings] = useState(1);
  const [stateName, setStateName] = useState('Delhi');
  
  // Response State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [saveMessage, setSaveMessage] = useState('');
  
  // History State
  const [history, setHistory] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    setSaveMessage('');

    const ingList = ingredients.split(',').map(i => i.trim()).filter(i => i);

    try {
      const response = await fetch('https://ratatouille-backend.onrender.com/generate-recipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ingredients: ingList,
          budget: parseFloat(budget),
          servings: parseInt(servings),
          state: stateName
        })
      });

      const data = await response.json();

      if (data.status === 'success') {
        setResult(data);
      } else {
        setError(data.message || 'Failed to generate recipe.');
      }
    } catch (err) {
      setError('Connection to backend failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveRecipe = async () => {
    if (!username) {
      alert("Please enter a username at the top right to save recipes!");
      return;
    }
    
    try {
      const response = await fetch('https://ratatouille-backend.onrender.com/save-recipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: username,
          recipe_data: result
        })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setSaveMessage('✅ Recipe saved successfully!');
      } else {
        alert(data.message);
      }
    } catch (err) {
      alert("Failed to connect to database.");
    }
  };

  const fetchHistory = async () => {
    if (!username) {
      alert("Please enter a username first!");
      return;
    }
    
    setLoading(true);
    setViewMode('history');
    
    try {
      const response = await fetch(`https://ratatouille-backend.onrender.com/my-recipes/${username}`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setHistory(data.recipes);
      } else {
        alert(data.message);
      }
    } catch (err) {
      alert("Failed to fetch history.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Top Navigation Bar */}
      <nav className="top-nav glass-panel fade-in">
        <div className="nav-left">
          <button className={`nav-tab ${viewMode === 'generate' ? 'active' : ''}`} onClick={() => setViewMode('generate')}>
            Cook
          </button>
          <button className={`nav-tab ${viewMode === 'history' ? 'active' : ''}`} onClick={fetchHistory}>
            My Recipes
          </button>
        </div>
        <div className="nav-right">
          <input 
            type="text" 
            placeholder="Enter Username to Save..." 
            value={username} 
            onChange={(e) => setUsername(e.target.value)} 
            className="username-input"
          />
        </div>
      </nav>

      {viewMode === 'generate' ? (
        <>
          <div className="glass-panel main-panel fade-in">
            <header>
              <h1 className="title">Ratatouille 👨‍🍳</h1>
              <p className="subtitle">AI-Powered Cost-Aware Recipe Generator</p>
            </header>

            <form onSubmit={handleSubmit} className="recipe-form">
              <div className="input-group">
                <label>Ingredients (comma separated)</label>
                <input 
                  type="text" 
                  value={ingredients} 
                  onChange={(e) => setIngredients(e.target.value)} 
                  required 
                  placeholder="e.g. paneer, tomato, onion"
                />
              </div>
              
              <div className="input-row">
                <div className="input-group">
                  <label>Budget (₹)</label>
                  <input 
                    type="number" 
                    value={budget} 
                    onChange={(e) => setBudget(e.target.value)} 
                    required 
                    min="10"
                  />
                </div>
                <div className="input-group">
                  <label>Servings</label>
                  <input 
                    type="number" 
                    value={servings} 
                    onChange={(e) => setServings(e.target.value)} 
                    required 
                    min="1"
                  />
                </div>
              </div>

              <div className="input-group">
                <label>Geographic State</label>
                <input 
                  type="text" 
                  value={stateName} 
                  onChange={(e) => setStateName(e.target.value)} 
                  required 
                  placeholder="e.g. Delhi, Maharashtra"
                />
              </div>

              <button type="submit" disabled={loading} className="generate-btn">
                {loading ? <span className="loader"></span> : 'Generate Recipe'}
              </button>
            </form>
          </div>

          {(result || error) && (
            <div className="glass-panel result-panel fade-in">
              {error ? (
                <div className="error-message">⚠️ {error}</div>
              ) : (
                <div className="recipe-content">
                  <div className="recipe-header">
                    <h2>{result.recipe.split('\n')[0].replace('### TITLE:', '').trim()}</h2>
                    <span className="badge">{result.archetype}</span>
                  </div>
                  
                  <div className="ingredients-box">
                    <h3>Calculated Market Ingredients</h3>
                    <ul>
                      {result.calculated_ingredients.map((ing, idx) => (
                        <li key={idx}>✓ {ing}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="directions-box">
                    <h3>Cooking Directions</h3>
                    <div className="recipe-text">
                      {result.recipe.includes('### DIRECTIONS:') 
                        ? result.recipe.split('### DIRECTIONS:')[1].trim().split('\n').map((step, idx) => (
                            <p key={idx} className="step-text">{step}</p>
                          ))
                        : <p className="step-text">{result.recipe}</p>}
                    </div>
                  </div>
                  
                  <div className="save-section">
                     {saveMessage ? (
                       <p className="save-success">{saveMessage}</p>
                     ) : (
                       <button className="save-btn" onClick={handleSaveRecipe}>💾 Save to My Account</button>
                     )}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        <div className="glass-panel main-panel fade-in">
          <header>
            <h1 className="title">My Recipes 📖</h1>
            <p className="subtitle">History for user: {username}</p>
          </header>
          
          {loading ? (
             <div className="loader" style={{margin: '0 auto'}}></div>
          ) : history.length === 0 ? (
             <p style={{textAlign: 'center', color: '#ccc'}}>No recipes saved yet!</p>
          ) : (
            <div className="history-list">
              {history.map((doc, idx) => {
                 const rec = doc.recipe;
                 return (
                   <div key={idx} className="history-card">
                     <h4>{rec.recipe.split('\n')[0].replace('### TITLE:', '').trim()}</h4>
                     <span className="badge">{rec.archetype}</span>
                     <p className="date-text">Saved: {new Date(doc.created_at * 1000).toLocaleString()}</p>
                   </div>
                 )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
