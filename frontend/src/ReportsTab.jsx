import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';

export default function ReportsTab({ backendUrl }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  const [yAxisSelection, setYAxisSelection] = useState('avgLatency');

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

  // Actual Pipeline names used in benchmark_groq.py
  const pipelines = ["Groq Bare", "Groq Budget", "Groq Full (Agentic)"];
  
  const stats = pipelines.map(pipe => {
    const pipeData = data.filter(d => d.Pipeline === pipe);
    const count = pipeData.length;
    
    // Parse CVS string to float (it might be "N/A" if it failed)
    const validCvsData = pipeData.filter(d => d["CVS Score"] !== "N/A" && typeof d["CVS Score"] === 'number');
    const avgLatency = count ? pipeData.reduce((acc, curr) => acc + (curr["Latency (sec)"] || 0), 0) / count : 0;
    const avgCVS = validCvsData.length ? validCvsData.reduce((acc, curr) => acc + curr["CVS Score"], 0) / validCvsData.length : 0;
    const totalCorrections = pipeData.reduce((acc, curr) => acc + (curr["Self_Correction_Attempts"] || 0), 0);
    const avgCorrections = count ? totalCorrections / count : 0;
    
    const budgetFailures = pipeData.filter(d => d["Budget Handled"] === "No" || d.Status !== "Success").length;
    const successRate = count ? ((count - budgetFailures) / count) * 100 : 0;
    
    return {
      name: pipe,
      avgLatency: parseFloat(avgLatency.toFixed(2)),
      avgCVS: parseFloat(avgCVS.toFixed(2)),
      avgCorrections: parseFloat(avgCorrections.toFixed(2)),
      successRate: parseFloat(successRate.toFixed(1)),
      totalRuns: count,
      budgetFailures: budgetFailures
    };
  });

  // Group data by Scenario for Side-by-Side Comparison
  const scenariosMap = {};
  data.forEach(row => {
    if (!scenariosMap[row.Scenario]) {
        scenariosMap[row.Scenario] = { 
            name: row.Scenario, 
            ingredients: row.Input_Ingredients,
            budget: row.Budget 
        };
    }
    scenariosMap[row.Scenario][row.Pipeline] = row;
  });
  const groupedScenarios = Object.values(scenariosMap);

  const yAxisConfig = {
      avgLatency: { label: "Latency (sec)", color: "#3b82f6", domain: ['auto', 'auto'] },
      avgCVS: { label: "Culinary Validity Score", color: "#10b981", domain: [0, 1] },
      avgCorrections: { label: "Avg Judge Loops", color: "#f59e0b", domain: ['auto', 'auto'] },
      successRate: { label: "Budget & Rules Success (%)", color: "#8b5cf6", domain: [0, 100] }
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '1rem', background: '#fff', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '2rem', color: '#1f2937' }}>Pipeline Evaluation Report</h2>
      
      <div style={{ background: '#f9fafb', padding: '1rem', borderRadius: '8px', border: '1px solid #e5e7eb', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ margin: 0, color: '#4b5563' }}>Performance Chart</h3>
            <div>
                <label style={{ marginRight: '10px', fontWeight: 'bold' }}>Select Y-Axis Metric:</label>
                <select 
                    value={yAxisSelection} 
                    onChange={(e) => setYAxisSelection(e.target.value)}
                    style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ccc' }}
                >
                    <option value="avgLatency">Average Latency (Seconds)</option>
                    <option value="avgCVS">Average CVS (Culinary Score)</option>
                    <option value="avgCorrections">Average Judge Loops</option>
                    <option value="successRate">Overall Success / Budget Handled (%)</option>
                </select>
            </div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={stats} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" />
            <YAxis domain={yAxisConfig[yAxisSelection].domain} />
            <Tooltip />
            <Bar dataKey={yAxisSelection} fill={yAxisConfig[yAxisSelection].color} radius={[4, 4, 0, 0]} name={yAxisConfig[yAxisSelection].label} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h3 style={{ borderBottom: '2px solid #e5e7eb', paddingBottom: '0.5rem', marginBottom: '1rem' }}>Side-by-Side Horizontal Comparison</h3>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {groupedScenarios.map((scenario, idx) => (
              <div key={idx} style={{ border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden' }}>
                  <div style={{ background: '#f3f4f6', padding: '1rem', borderBottom: '1px solid #e5e7eb' }}>
                      <strong>Scenario: {scenario.name}</strong> | Budget: ₹{scenario.budget} | Ingredients: {scenario.ingredients?.join(', ')}
                  </div>
                  <div style={{ display: 'flex', width: '100%' }}>
                      {pipelines.map((pipe) => {
                          const run = scenario[pipe];
                          return (
                              <div key={pipe} style={{ flex: 1, padding: '1rem', borderRight: '1px solid #e5e7eb', minWidth: '33%' }}>
                                  <h4 style={{ textAlign: 'center', color: '#374151', borderBottom: '1px dashed #ccc', paddingBottom: '0.5rem' }}>{pipe}</h4>
                                  {!run ? <p style={{ color: '#9ca3af' }}>No data for this scenario yet.</p> : (
                                      <div style={{ fontSize: '0.85rem' }}>
                                          <p><strong>Status:</strong> <span style={{ color: run.Status === 'Success' ? 'green' : 'red' }}>{run.Status}</span></p>
                                          <p><strong>Latency:</strong> {run["Latency (sec)"]}s</p>
                                          <p><strong>CVS Score:</strong> {run["CVS Score"]}</p>
                                          <p><strong>Judge Loops:</strong> {run.Self_Correction_Attempts}</p>
                                          
                                          {run.Calculated_Ingredients_Weights?.length > 0 && (
                                              <div style={{ marginTop: '0.5rem' }}>
                                                  <strong>Calculated Budgeted Weights:</strong>
                                                  <ul style={{ paddingLeft: '1.2rem', margin: '0.2rem 0', color: '#4b5563' }}>
                                                      {run.Calculated_Ingredients_Weights.map((w, i) => <li key={i}>{w}</li>)}
                                                  </ul>
                                              </div>
                                          )}

                                          {run.Judge_Critiques?.length > 0 && (
                                              <div style={{ marginTop: '0.5rem' }}>
                                                  <strong>Judge Critiques (Trace):</strong>
                                                  <ul style={{ paddingLeft: '1.2rem', margin: '0.2rem 0', color: '#b91c1c' }}>
                                                      {run.Judge_Critiques.map((c, i) => <li key={i}>{c}</li>)}
                                                  </ul>
                                              </div>
                                          )}

                                          <div style={{ marginTop: '1rem' }}>
                                              <strong>Generated Recipe:</strong>
                                              <div style={{ 
                                                  background: '#f9fafb', padding: '0.5rem', marginTop: '0.3rem', 
                                                  maxHeight: '300px', overflowY: 'auto', border: '1px solid #e5e7eb',
                                                  whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.8rem'
                                              }}>
                                                  {run.Generated_Recipe || "Failed to generate recipe."}
                                              </div>
                                          </div>
                                      </div>
                                  )}
                              </div>
                          )
                      })}
                  </div>
              </div>
          ))}
      </div>
    </div>
  );
}
