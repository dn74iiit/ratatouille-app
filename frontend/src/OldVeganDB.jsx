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