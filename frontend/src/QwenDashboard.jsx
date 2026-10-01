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
    <div className="glass-panel main-panel fade-in" style={{ padding: '2rem', background: '#ffffff', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)', borderRadius: '12px', border: '1px solid #f3f4f6' }}>
      <header style={{ marginBottom: '2rem', textAlign: 'center' }}>
        <h1 className="title" style={{ fontSize: '2.5rem', marginBottom: '0.5rem', color: 'var(--maroon)' }}>Agentic Comparison Dashboard 📊</h1>
        <p className="subtitle" style={{ color: 'var(--text-muted)' }}>Llama 3 (Fine-Tuned) vs Groq Llama/Mixtral (Generalist Zero-Shot)</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        <div style={{ background: '#f9fafb', padding: '1.5rem', borderRadius: '12px', textAlign: 'center', border: '1px solid #e5e7eb' }}>
          <h3 style={{ color: '#6b7280', fontSize: '0.9rem', textTransform: 'uppercase' }}>Recipes Processed</h3>
          <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--maroon)', margin: '0.5rem 0' }}>{total} / 144</p>
        </div>
        <div style={{ background: '#fef2f2', padding: '1.5rem', borderRadius: '12px', textAlign: 'center', borderTop: '4px solid #ef4444', borderBottom: '1px solid #e5e7eb', borderLeft: '1px solid #e5e7eb', borderRight: '1px solid #e5e7eb' }}>
          <h3 style={{ color: '#ef4444', fontSize: '0.9rem', textTransform: 'uppercase' }}>Llama 3 Avg Latency</h3>
          <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#dc2626', margin: '0.5rem 0' }}>{avgLlamaTime}s</p>
        </div>
        <div style={{ background: '#eff6ff', padding: '1.5rem', borderRadius: '12px', textAlign: 'center', borderTop: '4px solid #3b82f6', borderBottom: '1px solid #e5e7eb', borderLeft: '1px solid #e5e7eb', borderRight: '1px solid #e5e7eb' }}>
          <h3 style={{ color: '#3b82f6', fontSize: '0.9rem', textTransform: 'uppercase' }}>Qwen Avg Latency</h3>
          <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#2563eb', margin: '0.5rem 0' }}>{avgQwenTime}s</p>
        </div>
        <div style={{ background: '#f9fafb', padding: '1.5rem', borderRadius: '12px', textAlign: 'center', border: '1px solid #e5e7eb' }}>
          <h3 style={{ color: '#6b7280', fontSize: '0.9rem', textTransform: 'uppercase' }}>Total Hallucinations Caught</h3>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', marginTop: '0.5rem' }}>
            <div>
              <span style={{ color: '#ef4444', fontWeight: 'bold', fontSize: '1.2rem' }}>Llama:</span> <span style={{ color: '#374151', fontSize: '1.5rem', fontWeight: 'bold' }}>{llamaFails}</span>
            </div>
            <div>
              <span style={{ color: '#3b82f6', fontWeight: 'bold', fontSize: '1.2rem' }}>Qwen:</span> <span style={{ color: '#374151', fontSize: '1.5rem', fontWeight: 'bold' }}>{qwenFails}</span>
            </div>
          </div>
        </div>
      </div>

      <h2 style={{ color: 'var(--maroon)', marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem' }}>Side-by-Side Outputs</h2>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {data.slice().reverse().map((row, idx) => (
          <div key={idx} style={{ background: '#f9fafb', borderRadius: '12px', padding: '1.5rem', border: '1px solid #e5e7eb', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ color: 'var(--text-main)', fontSize: '1.3rem', margin: 0 }}>{row.original_title}</h3>
              <span className="badge">{row.archetype}</span>
            </div>
            <p style={{ color: '#6b7280', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
              <strong>Ingredients:</strong> {row.ingredients || 'Not recorded in dataset'}
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              {/* Llama Panel */}
              <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '8px', borderTop: '3px solid #ef4444', borderBottom: '1px solid #f3f4f6', borderLeft: '1px solid #f3f4f6', borderRight: '1px solid #f3f4f6' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid #f3f4f6', paddingBottom: '0.75rem', alignItems: 'center' }}>
                  <h4 style={{ color: '#ef4444', margin: 0, fontSize: '1.1rem' }}>Llama 3 V10</h4>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <span style={{ background: '#fef2f2', color: '#ef4444', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 'bold', border: '1px solid #fecaca' }}>⏱ {parseFloat(row.latency_sec || 0).toFixed(1)}s</span>
                    <span style={{ background: '#fef2f2', color: '#ef4444', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 'bold', border: '1px solid #fecaca' }}>🔄 {row.self_correction_attempts} Corrections</span>
                    <span style={{ background: '#fef2f2', color: '#ef4444', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 'bold', border: '1px solid #fecaca' }}>⭐ {row.final_cvs_score} Score</span>
                  </div>
                </div>
                <div style={{ whiteSpace: 'pre-wrap', color: '#374151', fontSize: '0.9rem', maxHeight: '300px', overflowY: 'auto', paddingRight: '0.5rem' }}>
                  {row.recipe}
                </div>
              </div>

              {/* Qwen Panel */}
              <div style={{ background: '#ffffff', padding: '1rem', borderRadius: '8px', borderTop: '3px solid #3b82f6', borderBottom: '1px solid #f3f4f6', borderLeft: '1px solid #f3f4f6', borderRight: '1px solid #f3f4f6' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid #f3f4f6', paddingBottom: '0.75rem', alignItems: 'center' }}>
                  <h4 style={{ color: '#3b82f6', margin: 0, fontSize: '1.1rem' }}>Groq Qwen 2.5</h4>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <span style={{ background: '#eff6ff', color: '#3b82f6', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 'bold', border: '1px solid #bfdbfe' }}>⏱ {parseFloat(row.qwen_latency_sec || 0).toFixed(1)}s</span>
                    <span style={{ background: '#eff6ff', color: '#3b82f6', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 'bold', border: '1px solid #bfdbfe' }}>🔄 {row.qwen_self_correction_attempts} Corrections</span>
                    <span style={{ background: '#eff6ff', color: '#3b82f6', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 'bold', border: '1px solid #bfdbfe' }}>⭐ {row.qwen_final_cvs_score} Score</span>
                  </div>
                </div>
                <div style={{ whiteSpace: 'pre-wrap', color: '#374151', fontSize: '0.9rem', maxHeight: '300px', overflowY: 'auto', paddingRight: '0.5rem' }}>
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
