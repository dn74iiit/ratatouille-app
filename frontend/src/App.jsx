import React, { useState } from 'react';
import './index.css';

function App() {
  const [ingredients, setIngredients] = useState('rice, egg, potato, tomato, onion');
  const [budget, setBudget] = useState(150);
  const [servings, setServings] = useState(1);
  const [stateName, setStateName] = useState('Delhi');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    const ingList = ingredients.split(',').map(i => i.trim()).filter(i => i);

    try {
      // Reverted to production URL for Vercel deployment
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
      setError('Connection to backend failed. Please try again. (Note: Render free tier takes 50 seconds to wake up if asleep)');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
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
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
