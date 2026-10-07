import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function ReportsTab({ backendUrl }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${backendUrl}/benchmark-reports`)
      .then(res => res.json())
      .then(result => {
        if (result.status === 'success') {
          setData(result.data);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [backendUrl]);

  if (loading) return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading benchmark data...</div>;
  if (!data || data.length === 0) return <div style={{ padding: '2rem', textAlign: 'center' }}>No benchmark data found. Run python benchmark_groq.py!</div>;

  // Aggregate Data
  const pipelines = ["Bare Qwen", "Budget Qwen", "Fast Qwen"]; // Fast Qwen = Full Agentic
  const stats = pipelines.map(pipe => {
    const pipeData = data.filter(d => d.Pipeline === pipe);
    const count = pipeData.length;
    const avgLatency = count ? pipeData.reduce((acc, curr) => acc + (curr["Latency (sec)"] || 0), 0) / count : 0;
    const avgCVS = count ? pipeData.reduce((acc, curr) => acc + (curr["CVS Score"] || 0), 0) / count : 0;
    const totalCorrections = pipeData.reduce((acc, curr) => acc + (curr["Self_Correction_Attempts"] || 0), 0);
    const avgCorrections = count ? totalCorrections / count : 0;
    
    // Calculate Budget Fail Rate (Where Budget Handled == 'No' or Status == 'Error')
    const budgetFailures = pipeData.filter(d => d["Budget Handled"] === "No" || d.Status === "Error").length;
    
    return {
      name: pipe.replace(' Qwen', ''),
      avgLatency: parseFloat(avgLatency.toFixed(2)),
      avgCVS: parseFloat(avgCVS.toFixed(2)),
      avgCorrections: parseFloat(avgCorrections.toFixed(2)),
      totalRuns: count,
      budgetFailures: budgetFailures
    };
  });

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '1rem', background: '#fff', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '2rem', color: '#1f2937' }}>Pipeline Evaluation Report</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginBottom: '3rem' }}>
        
        {/* Latency Chart */}
        <div style={{ background: '#f9fafb', padding: '1rem', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
          <h3 style={{ textAlign: 'center', marginBottom: '1rem', color: '#4b5563' }}>Average Latency (Seconds)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={stats} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="avgLatency" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Latency (s)" />
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: '0.85rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem' }}>Lower is better. Shows the "Cost of Agents".</p>
        </div>

        {/* CVS Score Chart */}
        <div style={{ background: '#f9fafb', padding: '1rem', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
          <h3 style={{ textAlign: 'center', marginBottom: '1rem', color: '#4b5563' }}>Average Culinary Validity Score (CVS)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={stats} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="avgCVS" fill="#10b981" radius={[4, 4, 0, 0]} name="CVS Score (0-1)" />
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: '0.85rem', color: '#6b7280', textAlign: 'center', marginTop: '0.5rem' }}>Higher is better. 1.0 means no culinary hallucinations.</p>
        </div>
      </div>

      <h3 style={{ borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Raw Data & Granular Step Times</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ background: '#f3f4f6', textAlign: 'left' }}>
              <th style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>Timestamp</th>
              <th style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>Pipeline</th>
              <th style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>Total Time</th>
              <th style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>SciPy Cost</th>
              <th style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>FAISS/Graph Cost</th>
              <th style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>Retries</th>
              <th style={{ padding: '0.75rem', borderBottom: '1px solid #e5e7eb' }}>CVS</th>
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 50).map((row, idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid #e5e7eb' }}>
                <td style={{ padding: '0.75rem' }}>{new Date(row.Timestamp * 1000).toLocaleTimeString()}</td>
                <td style={{ padding: '0.75rem', fontWeight: '500' }}>{row.Pipeline.replace(' Qwen', '')}</td>
                <td style={{ padding: '0.75rem' }}>{(row["Latency (sec)"] || 0).toFixed(2)}s</td>
                <td style={{ padding: '0.75rem' }}>{row.Granular_Step_Times?.optimization_sec ? `${row.Granular_Step_Times.optimization_sec.toFixed(3)}s` : '-'}</td>
                <td style={{ padding: '0.75rem' }}>{row.Granular_Step_Times?.retrieval_sec ? `${row.Granular_Step_Times.retrieval_sec.toFixed(3)}s` : '-'}</td>
                <td style={{ padding: '0.75rem' }}>{row.Self_Correction_Attempts || 0}</td>
                <td style={{ padding: '0.75rem', color: row["CVS Score"] < 0.8 ? '#ef4444' : '#10b981' }}>{row["CVS Score"]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data.length > 50 && <p style={{ textAlign: 'center', marginTop: '1rem', color: '#6b7280', fontSize: '0.85rem' }}>Showing latest 50 records...</p>}
      </div>
    </div>
  );
}
