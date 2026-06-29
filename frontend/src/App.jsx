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
  const [isVegan, setIsVegan] = useState(false);
  
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

  // Vegan DB State
  const [dbIngredient, setDbIngredient] = useState('');
  const [dbAlternatives, setDbAlternatives] = useState(null);
  const [dbAltLoading, setDbAltLoading] = useState(false);
  const [dbAltError, setDbAltError] = useState('');
  const [dbAltSource, setDbAltSource] = useState('');
  const [indianRecipes, setIndianRecipes] = useState([]);
  const [indianStyle, setIndianStyle] = useState('');
  const [indianLoading, setIndianLoading] = useState(false);
  const [indianTotal, setIndianTotal] = useState(0);
  const [expandedRecipe, setExpandedRecipe] = useState(null);


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
          model_version: modelVersion,
          is_vegan: isVegan
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
          <button className={`nav-tab ${viewMode === 'vegandb' ? 'active' : ''}`} onClick={() => setViewMode('vegandb')}>
            Vegan DB 🥦
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

              {/* Vegan Toggle */}
              <div className="input-group" style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontSize: '1.2rem', color: '#10b981', fontWeight: 'bold' }}>
                  <input 
                    type="checkbox" 
                    checked={isVegan} 
                    onChange={(e) => setIsVegan(e.target.checked)} 
                    style={{ width: '24px', height: '24px', cursor: 'pointer', accentColor: '#10b981' }}
                  />
                  🌿 Make it Vegan
                </label>
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

      {/* ── Vegan DB Panel ────────────────────────────────────────────────────── */}
      {viewMode === 'vegandb' && (
        <div className="glass-panel main-panel fade-in">
          <header>
            <h1 className="title">Vegan Database 🥦</h1>
            <p className="subtitle">GPU-generated vegan alternatives &amp; Indian budget recipes</p>
          </header>

          {/* ── Panel A: Ingredient Lookup ── */}
          <section style={{marginBottom:'2rem'}}>
            <h2 style={{color:'var(--accent)',marginBottom:'0.75rem'}}>🔍 Ingredient Lookup</h2>
            <p style={{color:'var(--text-muted)',marginBottom:'1rem',fontSize:'0.9rem'}}>
              Type any non-veg ingredient to see its top-5 vegan substitutes with chemical scores &amp; spice bridges.
            </p>
            <div style={{display:'flex',gap:'0.75rem',flexWrap:'wrap'}}>
              <input
                type="text"
                value={dbIngredient}
                onChange={e => setDbIngredient(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    if (!dbIngredient.trim()) return;
                    setDbAltLoading(true); setDbAltError(''); setDbAlternatives(null); setDbAltSource('');
                    fetch(`${BACKEND_URL}/vegan-alternatives/${encodeURIComponent(dbIngredient.trim().toLowerCase())}`)
                      .then(r => r.json())
                      .then(d => {
                        if (d.status === 'success') { setDbAlternatives(d.alternatives); setDbAltSource(d.source); }
                        else if (d.status === 'already_vegan') setDbAltError('✅ ' + d.message);
                        else setDbAltError(d.message || 'Not found.');
                      })
                      .catch(() => setDbAltError('Connection failed.'))
                      .finally(() => setDbAltLoading(false));
                  }
                }}
                placeholder="e.g. chicken, mutton, egg, shrimp…"
                style={{flex:'1',minWidth:'200px'}}
              />
              <button
                className="btn-primary"
                disabled={dbAltLoading || !dbIngredient.trim()}
                onClick={() => {
                  if (!dbIngredient.trim()) return;
                  setDbAltLoading(true); setDbAltError(''); setDbAlternatives(null); setDbAltSource('');
                  fetch(`${BACKEND_URL}/vegan-alternatives/${encodeURIComponent(dbIngredient.trim().toLowerCase())}`)
                    .then(r => r.json())
                    .then(d => {
                      if (d.status === 'success') { setDbAlternatives(d.alternatives); setDbAltSource(d.source); }
                      else if (d.status === 'already_vegan') setDbAltError('✅ ' + d.message);
                      else setDbAltError(d.message || 'Not found.');
                    })
                    .catch(() => setDbAltError('Connection failed.'))
                    .finally(() => setDbAltLoading(false));
                }}
              >{dbAltLoading ? 'Searching…' : 'Search'}</button>
            </div>

            {dbAltError && <p style={{color:'#f97316',marginTop:'0.75rem'}}>{dbAltError}</p>}

            {dbAlternatives && (
              <div style={{marginTop:'1rem'}}>
                <div style={{display:'flex',alignItems:'center',gap:'0.5rem',marginBottom:'0.75rem'}}>
                  <span style={{fontSize:'0.85rem',color:'var(--text-muted)'}}>Source:</span>
                  <span style={{
                    background: dbAltSource === 'db' ? 'rgba(34,197,94,0.15)' : 'rgba(249,115,22,0.15)',
                    color: dbAltSource === 'db' ? '#22c55e' : '#f97316',
                    borderRadius:'999px', padding:'2px 10px', fontSize:'0.78rem', fontWeight:600
                  }}>
                    {dbAltSource === 'db' ? '⚡ GPU-generated DB' : '🔄 Live Engine'}
                  </span>
                </div>
                {dbAlternatives.map((alt, idx) => (
                  <div key={idx} className="glass-panel" style={{padding:'1rem',marginBottom:'0.75rem',borderLeft:'3px solid var(--accent)'}}>
                    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',flexWrap:'wrap',gap:'0.5rem'}}>
                      <div style={{display:'flex',alignItems:'center',gap:'0.6rem'}}>
                        <span style={{fontWeight:700,fontSize:'1.05rem',color:'var(--text-primary)'}}>
                          #{alt.rank} {alt.substitute}
                        </span>
                      </div>
                      <span style={{
                        background:'rgba(99,102,241,0.15)',color:'#818cf8',
                        borderRadius:'999px',padding:'3px 12px',fontWeight:700,fontSize:'0.85rem'
                      }}>
                        Score: {(alt.composite_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    {alt.score_breakdown && (
                      <div style={{display:'flex',gap:'0.5rem',flexWrap:'wrap',marginTop:'0.5rem'}}>
                        {Object.entries(alt.score_breakdown).map(([k,v]) => (
                          <span key={k} style={{
                            background:'rgba(255,255,255,0.06)',borderRadius:'6px',
                            padding:'2px 8px',fontSize:'0.78rem',color:'var(--text-muted)'
                          }}>{k.replace('_',' ')}: {(v*100).toFixed(0)}%</span>
                        ))}
                      </div>
                    )}
                    {alt.spice_bridge && alt.spice_bridge.length > 0 && (
                      <div style={{marginTop:'0.6rem'}}>
                        <span style={{fontSize:'0.8rem',color:'var(--text-muted)'}}>🌶 Spice Bridge: </span>
                        {alt.spice_bridge.map((s,si) => (
                          <span key={si} style={{
                            background:'rgba(251,191,36,0.12)',color:'#fbbf24',
                            borderRadius:'999px',padding:'2px 8px',fontSize:'0.78rem',
                            marginLeft:'4px',display:'inline-block'
                          }}>{s.spice || s}</span>
                        ))}
                      </div>
                    )}
                    {alt.culinary_notes && (
                      <p style={{marginTop:'0.5rem',fontSize:'0.84rem',color:'var(--text-muted)',fontStyle:'italic'}}>
                        💡 {alt.culinary_notes}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>

          <hr style={{border:'none',borderTop:'1px solid rgba(255,255,255,0.08)',marginBottom:'2rem'}}/>

          {/* ── Panel B: Browse Indian Recipes ── */}
          <section>
            <h2 style={{color:'var(--accent)',marginBottom:'0.75rem'}}>🇮🇳 Browse Indian Budget Vegan Recipes</h2>
            <p style={{color:'var(--text-muted)',marginBottom:'1rem',fontSize:'0.9rem'}}>
              GPU-generated recipes under ₹150/serving using your V10 model.
            </p>
            <div style={{display:'flex',gap:'0.75rem',flexWrap:'wrap',alignItems:'center',marginBottom:'1rem'}}>
              <select
                value={indianStyle}
                onChange={e => setIndianStyle(e.target.value)}
                style={{padding:'0.5rem 0.75rem',background:'rgba(255,255,255,0.05)',
                  border:'1px solid rgba(255,255,255,0.1)',borderRadius:'8px',color:'var(--text-primary)'}}
              >
                <option value="">All Styles</option>
                <option value="Curry">Curry</option>
                <option value="Dry_Sabzi">Dry Sabzi</option>
                <option value="Rice_Dish">Rice Dish</option>
                <option value="Soup">Soup (Dal)</option>
              </select>
              <button
                className="btn-primary"
                disabled={indianLoading}
                onClick={() => {
                  setIndianLoading(true);
                  const q = indianStyle ? `?style=${encodeURIComponent(indianStyle)}&limit=12` : '?limit=12';
                  fetch(`${BACKEND_URL}/indian-recipes${q}`)
                    .then(r => r.json())
                    .then(d => {
                      if (d.status === 'success') { setIndianRecipes(d.recipes); setIndianTotal(d.total); }
                    })
                    .catch(() => {})
                    .finally(() => setIndianLoading(false));
                }}
              >{indianLoading ? 'Loading…' : 'Load Recipes'}</button>
              {indianTotal > 0 && <span style={{color:'var(--text-muted)',fontSize:'0.85rem'}}>{indianTotal} recipes total</span>}
            </div>

            <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(300px,1fr))',gap:'1rem'}}>
              {indianRecipes.map((r, i) => (
                <div key={r._id || i} className="glass-panel" style={{padding:'1.1rem',display:'flex',flexDirection:'column',gap:'0.5rem'}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',gap:'0.5rem'}}>
                    <h3 style={{margin:0,fontSize:'1rem',color:'var(--text-primary)',flex:1}}>{r.dish_name}</h3>
                    <span style={{
                      background:'rgba(34,197,94,0.15)',color:'#22c55e',
                      borderRadius:'999px',padding:'2px 10px',fontSize:'0.75rem',fontWeight:700,whiteSpace:'nowrap'
                    }}>₹{r.budget_inr}</span>
                  </div>
                  <div style={{display:'flex',gap:'0.4rem',flexWrap:'wrap'}}>
                    <span style={{
                      background:'rgba(99,102,241,0.15)',color:'#818cf8',
                      borderRadius:'6px',padding:'1px 8px',fontSize:'0.75rem'
                    }}>{r.style}</span>
                    {r.original_non_veg && r.original_non_veg !== 'vegan' && (
                      <span style={{
                        background:'rgba(249,115,22,0.12)',color:'#f97316',
                        borderRadius:'6px',padding:'1px 8px',fontSize:'0.75rem'
                      }}>was: {r.original_non_veg}</span>
                    )}
                    <span style={{
                      background:'rgba(34,197,94,0.1)',color:'#4ade80',
                      borderRadius:'6px',padding:'1px 8px',fontSize:'0.75rem'
                    }}>🌿 {r.vegan_substitute}</span>
                  </div>
                  {r.ingredients_list && (
                    <p style={{margin:0,fontSize:'0.78rem',color:'var(--text-muted)',lineHeight:1.4}}>
                      {(Array.isArray(r.ingredients_list) ? r.ingredients_list : [r.ingredients_list]).slice(0,6).join(', ')}
                      {(Array.isArray(r.ingredients_list) ? r.ingredients_list.length : 0) > 6 ? '…' : ''}
                    </p>
                  )}
                  <button
                    style={{
                      marginTop:'auto',padding:'0.35rem 0.75rem',
                      background:'rgba(99,102,241,0.15)',border:'1px solid rgba(99,102,241,0.3)',
                      borderRadius:'8px',color:'#818cf8',cursor:'pointer',fontSize:'0.82rem'
                    }}
                    onClick={() => setExpandedRecipe(expandedRecipe === (r._id||i) ? null : (r._id||i))}
                  >{expandedRecipe === (r._id||i) ? 'Hide Recipe ▲' : 'View Recipe ▼'}</button>
                  {expandedRecipe === (r._id||i) && (
                    <pre style={{
                      marginTop:'0.5rem',background:'rgba(0,0,0,0.3)',borderRadius:'8px',
                      padding:'0.75rem',fontSize:'0.78rem',color:'var(--text-muted)',
                      whiteSpace:'pre-wrap',wordBreak:'break-word',maxHeight:'300px',overflowY:'auto'
                    }}>{r.generated_recipe}</pre>
                  )}
                </div>
              ))}
              {indianRecipes.length === 0 && !indianLoading && (
                <p style={{color:'var(--text-muted)',gridColumn:'1/-1',textAlign:'center',paddingTop:'2rem'}}>
                  Click "Load Recipes" to fetch from the database.
                </p>
              )}
            </div>
          </section>
        </div>
      )}

    </div>
  );
}

export default App;
