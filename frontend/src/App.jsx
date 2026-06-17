import React, { useState } from 'react';
import './index.css';

const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000' : 'https://ratatouille-backend.onrender.com';

function App() {
  const [viewMode, setViewMode] = useState('generate'); // 'generate' | 'vegan' | 'history'
  const [username, setUsername] = useState('');
  
  // Form State
  const [ingredients, setIngredients] = useState('rice, egg, potato, tomato, onion');
  const [budget, setBudget] = useState(150);
  const [servings, setServings] = useState(1);
  const [stateName, setStateName] = useState('Delhi');
  const [modelVersion, setModelVersion] = useState('v10');  // 'v8' | 'v10'
  
  // Response State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [saveMessage, setSaveMessage] = useState('');
  
  // History State
  const [history, setHistory] = useState([]);

  // Vegan Engine State
  const [veganIngredients, setVeganIngredients] = useState('paneer, chicken');
  const [veganArchetype, setVeganArchetype] = useState('Curry');
  const [veganResult, setVeganResult] = useState(null);
  const [veganLoading, setVeganLoading] = useState(false);
  const [veganError, setVeganError] = useState('');


  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    setSaveMessage('');

    const ingList = ingredients.split(',').map(i => i.trim()).filter(i => i);

    try {
      const response = await fetch(`${BACKEND_URL}/generate-recipe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ingredients: ingList,
          budget: parseFloat(budget),
          servings: parseInt(servings),
          state: stateName,
          model_version: modelVersion
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
      const response = await fetch(`${BACKEND_URL}/save-recipe`, {
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
      const response = await fetch(`${BACKEND_URL}/my-recipes/${username}`);
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

  const handleVeganConvert = async (e) => {
    e.preventDefault();
    setVeganLoading(true);
    setVeganError('');
    setVeganResult(null);

    const list = veganIngredients.split(',').map(i => i.trim()).filter(i => i);

    try {
      const response = await fetch(`${BACKEND_URL}/get-vegan-blueprint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ingredients: list,
          archetype: veganArchetype
        })
      });

      const data = await response.json();

      if (data.status === 'success') {
        setVeganResult(data.results);
      } else {
        setVeganError(data.message || 'Failed to generate vegan blueprint.');
      }
    } catch (err) {
      setVeganError('Connection to backend failed. Please make sure local api server is running.');
    } finally {
      setVeganLoading(false);
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
          <button className={`nav-tab ${viewMode === 'vegan' ? 'active' : ''}`} onClick={() => setViewMode('vegan')}>
            Vegan Converter 🌿
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

      {viewMode === 'generate' && (
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

              {/* Model Version Toggle */}
              <div className="input-group">
                <label>AI Model</label>
                <div className="model-toggle">
                  <button
                    type="button"
                    className={`model-btn ${modelVersion === 'v8' ? 'active' : ''}`}
                    onClick={() => setModelVersion('v8')}
                  >
                    V8 <span className="model-tag">Original</span>
                  </button>
                  <button
                    type="button"
                    className={`model-btn ${modelVersion === 'v10' ? 'active' : ''}`}
                    onClick={() => setModelVersion('v10')}
                  >
                    V10 <span className="model-tag">RecipeDB Retrained ✨</span>
                  </button>
                </div>
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
      )}

      {viewMode === 'vegan' && (
        <>
          <div className="glass-panel main-panel fade-in">
            <header>
              <h1 className="title">Vegan Substitution Engine 🌿</h1>
              <p className="subtitle">Find the best physical-chemical plant alternatives</p>
            </header>

            <form onSubmit={handleVeganConvert} className="recipe-form">
              <div className="input-group">
                <label>Ingredients to Convert (comma separated)</label>
                <input 
                  type="text" 
                  value={veganIngredients} 
                  onChange={(e) => setVeganIngredients(e.target.value)} 
                  required 
                  placeholder="e.g. paneer, chicken, butter, egg"
                />
              </div>

              <div className="input-group">
                <label>Dish Archetype (determines technique context)</label>
                <select 
                  value={veganArchetype} 
                  onChange={(e) => setVeganArchetype(e.target.value)}
                  className="username-input"
                  style={{ width: '100%', padding: '0.8rem', background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#fff', fontSize: '1rem', outline: 'none' }}
                >
                  <option value="Curry" style={{ background: '#1f2937' }}>Curry</option>
                  <option value="Dry_Sabzi" style={{ background: '#1f2937' }}>Dry Sabzi</option>
                  <option value="Salad" style={{ background: '#1f2937' }}>Salad</option>
                  <option value="Soup" style={{ background: '#1f2937' }}>Soup</option>
                  <option value="Dessert" style={{ background: '#1f2937' }}>Dessert</option>
                  <option value="Bread" style={{ background: '#1f2937' }}>Bread</option>
                  <option value="Rice_Dish" style={{ background: '#1f2937' }}>Rice Dish</option>
                </select>
              </div>

              <button type="submit" disabled={veganLoading} className="generate-btn">
                {veganLoading ? <span className="loader"></span> : 'Calculate Vegan Blueprint'}
              </button>
            </form>
          </div>

          {(veganResult || veganError) && (
            <div className="glass-panel result-panel fade-in">
              {veganError ? (
                <div className="error-message">⚠️ {veganError}</div>
              ) : (
                <div className="vegan-blueprint-results">
                  <h2 style={{ fontSize: '1.6rem', marginBottom: '1.5rem', textAlign: 'center', color: '#10b981' }}>
                    🌿 Calculated Substitution Blueprint
                  </h2>
                  {veganResult.map((res, idx) => (
                    <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', padding: '1.5rem', borderRadius: '12px', marginBottom: '1.5rem', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      {res.status === 'already_vegan' ? (
                        <div style={{ textAlign: 'center', color: '#10B981' }}>
                          <h3 style={{ margin: '0 0 0.5rem 0' }}>✓ {(res.original_ingredient || '').toUpperCase()}</h3>
                          <p style={{ margin: 0, opacity: 0.8 }}>This ingredient is already vegan!</p>
                        </div>
                      ) : res.status === 'error' ? (
                        <div style={{ color: '#EF4444' }}>
                          <h3 style={{ margin: '0 0 0.5rem 0' }}>⚠️ {(res.original_ingredient || '').toUpperCase()}</h3>
                          <p style={{ margin: 0, opacity: 0.8 }}>{res.message}</p>
                        </div>
                      ) : (
                        <div>
                          {/* Header */}
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.8rem', marginBottom: '1rem' }}>
                            <div>
                              <span style={{ fontSize: '0.8rem', opacity: 0.5 }}>Substitution:</span>
                              <h3 style={{ margin: 0, textTransform: 'capitalize', color: '#f3f4f6', fontSize: '1.2rem' }}>
                                {res.original_ingredient} → <span style={{ color: '#10b981' }}>{res.best_vegan_substitute}</span>
                              </h3>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                              <span style={{ fontSize: '0.8rem', opacity: 0.5 }}>Match Score:</span>
                              <h4 style={{ margin: 0, color: '#10b981', fontSize: '1.2rem' }}>{(res.match_score * 100).toFixed(0)}%</h4>
                            </div>
                          </div>

                          {/* Scores Breakdown */}
                          <div style={{ marginBottom: '1.2rem' }}>
                            <h4 style={{ margin: '0 0 0.6rem 0', fontSize: '0.9rem', opacity: 0.8 }}>Similarity Breakdown:</h4>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                              <div>
                                <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>Aroma/Flavor Fit</span>
                                <div style={{ background: '#374151', height: '6px', borderRadius: '3px', marginTop: '4px' }}>
                                  <div style={{ width: `${res.score_breakdown.flavor_similarity * 100}%`, background: '#3b82f6', height: '100%', borderRadius: '3px' }}></div>
                                </div>
                                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>{(res.score_breakdown.flavor_similarity * 100).toFixed(0)}%</span>
                              </div>
                              <div>
                                <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>Textural Similarity</span>
                                <div style={{ background: '#374151', height: '6px', borderRadius: '3px', marginTop: '4px' }}>
                                  <div style={{ width: `${res.score_breakdown.texture_similarity * 100}%`, background: '#8b5cf6', height: '100%', borderRadius: '3px' }}></div>
                                </div>
                                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>{(res.score_breakdown.texture_similarity * 100).toFixed(0)}%</span>
                              </div>
                              <div>
                                <span style={{ fontSize: '0.75rem', opacity: 0.6 }}>Culinary Role Match</span>
                                <div style={{ background: '#374151', height: '6px', borderRadius: '3px', marginTop: '4px' }}>
                                  <div style={{ width: `${res.score_breakdown.functional_fit * 100}%`, background: '#10b981', height: '100%', borderRadius: '3px' }}></div>
                                </div>
                                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>{(res.score_breakdown.functional_fit * 100).toFixed(0)}%</span>
                              </div>
                            </div>
                          </div>

                          {/* Compensation Blueprint */}
                          <div style={{ background: 'rgba(0,0,0,0.15)', padding: '1rem', borderRadius: '8px' }}>
                            <h4 style={{ margin: '0 0 0.8rem 0', color: '#10b981', fontSize: '0.95rem' }}>Delta Compensation Steps:</h4>
                            
                            {/* Prep Techniques */}
                            {res.compensation_blueprint.techniques.length > 0 && (
                              <div style={{ marginBottom: '1rem' }}>
                                <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.8rem', color: '#9ca3af' }}>Physical & Textural Prep:</h5>
                                <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.85rem', color: '#d1d5db' }}>
                                  {res.compensation_blueprint.techniques.map((tech, tIdx) => (
                                    <li key={tIdx} style={{ marginBottom: '0.3rem' }}>{tech}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Auxiliary Additions */}
                            {res.compensation_blueprint.auxiliary_additions.length > 0 && (
                              <div style={{ marginBottom: '1rem' }}>
                                <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.8rem', color: '#9ca3af' }}>Recommended Compound Additions:</h5>
                                <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.85rem', color: '#d1d5db' }}>
                                  {res.compensation_blueprint.auxiliary_additions.map((add, aIdx) => (
                                    <li key={aIdx} style={{ marginBottom: '0.3rem' }}>
                                      <strong>{add.amount} {add.name}</strong> - <em>{add.purpose}</em>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Spice Bridges */}
                            {res.compensation_blueprint.spice_bridge.length > 0 && (
                              <div>
                                <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.8rem', color: '#9ca3af' }}>Aromatic Spice Bridges:</h5>
                                <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.85rem', color: '#d1d5db' }}>
                                  {res.compensation_blueprint.spice_bridge.map((spice, sIdx) => (
                                    <li key={sIdx} style={{ marginBottom: '0.3rem' }}>
                                      Add <strong>{spice.spice}</strong> (covers {(spice.fills_gap_ratio * 100).toFixed(0)}% gap: <em>{spice.reason}</em>)
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}

      {viewMode === 'history' && (
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
                   <details key={idx} className="history-card">
                     <summary className="history-summary">
                       <div>
                         <h4>{rec.recipe.split('\n')[0].replace('### TITLE:', '').trim()}</h4>
                         <span className="badge">{rec.archetype}</span>
                         <p className="date-text">Saved: {new Date(doc.created_at * 1000).toLocaleString()}</p>
                       </div>
                       <div className="expand-hint">Click to view ▾</div>
                     </summary>
                     
                     <div className="history-expanded-content">
                        <div className="ingredients-box" style={{marginTop: '1rem'}}>
                          <h4>Ingredients</h4>
                          <ul>
                            {rec.calculated_ingredients.map((ing, i) => (
                              <li key={i}>✓ {ing}</li>
                            ))}
                          </ul>
                        </div>
                        
                        <div className="directions-box" style={{marginTop: '1rem'}}>
                          <h4>Directions</h4>
                          <div className="recipe-text" style={{fontSize: '0.95rem'}}>
                            {rec.recipe.includes('### DIRECTIONS:') 
                              ? rec.recipe.split('### DIRECTIONS:')[1].trim().split('\n').map((step, i) => (
                                  <p key={i} className="step-text" style={{padding: '0.8rem', marginBottom: '0.5rem'}}>{step}</p>
                                ))
                              : <p className="step-text">{rec.recipe}</p>}
                          </div>
                        </div>
                     </div>
                   </details>
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
