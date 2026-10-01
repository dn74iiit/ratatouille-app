import React, { useState, useEffect } from 'react';
import Papa from 'https://cdn.jsdelivr.net/npm/papaparse@5.4.1/+esm';

export default function QwenDashboard() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchCSV = async () => {
    try {
      const res = await fetch('/Reference_Catalog_v4_qwen_comparison.csv');
      if (!res.ok) {
        if (res.status === 404) {
          setData([]);
          setLoading(false);
          return;
        }
        throw new Error('Failed to fetch CSV');
      }
      const csvText = await res.text();
      
      Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          setData(results.data);
          setLoading(false);
        }
      });
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCSV();
  }, []);

  if (loading) {
    return (
      <div className="glass-panel main-panel fade-in" style={{ textAlign: 'center', padding: '3rem' }}>
        <h2 style={{color: 'white'}}>Loading Comparison Dashboard...</h2>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="glass-panel main-panel fade-in" style={{ textAlign: 'center', padding: '3rem' }}>
        <h2 style={{color: 'white'}}>Waiting for Qwen script to start...</h2>
        <p style={{color: '#ccc'}}>The dashboard will automatically update once the first recipe is saved to the CSV.</p>
      </div>
    );
  }

  // Calculate Aggregates
  const total = data.length;
  let qwenFails = 0;
  let llamaFails = 0;
  let qwenTime = 0;
  let llamaTime = 0;

  data.forEach(row => {
    qwenFails += parseInt(row.qwen_self_correction_attempts || 0);
    llamaFails += parseInt(row.self_correction_attempts || 0);
    qwenTime += parseFloat(row.qwen_latency_sec || 0);
    llamaTime += parseFloat(row.latency_sec || 0);
  });

  const avgQwenTime = (qwenTime / total).toFixed(2);
  const avgLlamaTime = (llamaTime / total).toFixed(2);

  return (
    <div className="glass-panel main-panel fade-in" style={{ padding: '2rem' }}>
      <header style={{ marginBottom: '2rem', textAlign: 'center' }}>
        <h1 className="title" style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Agentic Comparison Dashboard 📊</h1>
        <p className="subtitle">Llama 3 (Fine-Tuned) vs Groq Llama/Mixtral (Generalist Zero-Shot)</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1.5rem', borderRadius: '12px', textAlign: 'center' }}>
          <h3 style={{ color: '#9ca3af', fontSize: '0.9rem', textTransform: 'uppercase' }}>Recipes Processed</h3>
          <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#fff', margin: '0.5rem 0' }}>{total} / 144</p>
        </div>
        <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1.5rem', borderRadius: '12px', textAlign: 'center', borderTop: '4px solid #ef4444' }}>
          <h3 style={{ color: '#9ca3af', fontSize: '0.9rem', textTransform: 'uppercase' }}>Llama 3 Avg Latency</h3>
          <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#fff', margin: '0.5rem 0' }}>{avgLlamaTime}s</p>
        </div>
        <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1.5rem', borderRadius: '12px', textAlign: 'center', borderTop: '4px solid #3b82f6' }}>
          <h3 style={{ color: '#9ca3af', fontSize: '0.9rem', textTransform: 'uppercase' }}>Qwen Avg Latency</h3>
          <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#fff', margin: '0.5rem 0' }}>{avgQwenTime}s</p>
        </div>
        <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1.5rem', borderRadius: '12px', textAlign: 'center' }}>
          <h3 style={{ color: '#9ca3af', fontSize: '0.9rem', textTransform: 'uppercase' }}>Total Hallucinations Caught</h3>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', marginTop: '0.5rem' }}>
            <div>
              <span style={{ color: '#ef4444', fontWeight: 'bold', fontSize: '1.2rem' }}>Llama:</span> <span style={{ color: '#fff', fontSize: '1.5rem', fontWeight: 'bold' }}>{llamaFails}</span>
            </div>
            <div>
              <span style={{ color: '#3b82f6', fontWeight: 'bold', fontSize: '1.2rem' }}>Qwen:</span> <span style={{ color: '#fff', fontSize: '1.5rem', fontWeight: 'bold' }}>{qwenFails}</span>
            </div>
          </div>
        </div>
      </div>

      <h2 style={{ color: 'white', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>Side-by-Side Outputs</h2>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {data.slice().reverse().map((row, idx) => (
          <div key={idx} style={{ background: 'rgba(0,0,0,0.4)', borderRadius: '12px', padding: '1.5rem', border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ color: 'white', fontSize: '1.3rem', margin: 0 }}>{row.original_title}</h3>
              <span className="badge">{row.archetype}</span>
            </div>
            <p style={{ color: '#9ca3af', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
              <strong>Ingredients:</strong> {row.ingredients}
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              {/* Llama Panel */}
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '8px', borderTop: '3px solid #ef4444' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>
                  <h4 style={{ color: '#ef4444', margin: 0 }}>Llama 3 V10</h4>
                  <div style={{ fontSize: '0.85rem', color: '#9ca3af', textAlign: 'right' }}>
                    <div>Time: <strong>{row.latency_sec}s</strong></div>
                    <div>Corrections: <strong>{row.self_correction_attempts}</strong></div>
                    <div>Score: <strong>{row.final_cvs_score}</strong></div>
                  </div>
                </div>
                <div style={{ whiteSpace: 'pre-wrap', color: '#e5e7eb', fontSize: '0.9rem', maxHeight: '300px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                  {row.recipe}
                </div>
              </div>

              {/* Qwen Panel */}
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '8px', borderTop: '3px solid #3b82f6' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>
                  <h4 style={{ color: '#3b82f6', margin: 0 }}>Groq Qwen 2.5</h4>
                  <div style={{ fontSize: '0.85rem', color: '#9ca3af', textAlign: 'right' }}>
                    <div>Time: <strong>{row.qwen_latency_sec}s</strong></div>
                    <div>Corrections: <strong>{row.qwen_self_correction_attempts}</strong></div>
                    <div>Score: <strong>{row.qwen_final_cvs_score}</strong></div>
                  </div>
                </div>
                <div style={{ whiteSpace: 'pre-wrap', color: '#e5e7eb', fontSize: '0.9rem', maxHeight: '300px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                  {row.qwen_recipe}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
