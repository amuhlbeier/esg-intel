import { useState, useEffect } from 'react';
import peerData from './peerData.json';

const defaultCompany = 'AMZN';

export default function App() {
  const [company, setCompany] = useState(defaultCompany);
  const [analysis, setAnalysis] = useState(null);
  const [peers, setPeers] = useState([]);
  const [industryFilter, setIndustryFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPeers = async () => {
    const params = new URLSearchParams();
    if (industryFilter) params.set('industry', industryFilter);
    params.set('limit', '6');
    const resp = await fetch(`/api/peers?${params.toString()}`);
    if (resp.ok) {
      const data = await resp.json();
      setPeers(data.peers || []);
    }
  };

  useEffect(() => {
    fetchPeers().catch(() => setPeers([]));
  }, []);

  const runAnalysis = async () => {
    if (!company) return;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ company });
      const resp = await fetch(`/api/analyze?${params.toString()}`);
      if (!resp.ok) {
        throw new Error(`API error ${resp.status}`);
      }
      const data = await resp.json();
      setAnalysis(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch analysis');
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  const summaryMetrics = analysis?.evaluation?.evaluation?.metrics;

  return (
    <div className="app-shell">
      <header>
        <h1>ESG Intel</h1>
      </header>

      <div className="card" style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
        <div style={{ flex: 1 }}>
          <label htmlFor="company-select">Company ticker</label>
          <select
            id="company-select"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
          >
            {peerData.tickers.map((ticker) => (
              <option key={ticker} value={ticker}>
                {ticker}
              </option>
            ))}
          </select>
        </div>
        <button onClick={runAnalysis} disabled={loading || !company}>
          {loading ? 'Running…' : 'Run Analysis'}
        </button>
      </div>

      {error && <div className="card" style={{ color: '#dc2626' }}>{error}</div>}

      {analysis && (
        <section className="card">
          <h2>{analysis.company} ESG Summary</h2>
          <div>
            <h3>Environment</h3>
            <ul>
              {(analysis.summary?.environment || []).map((item, idx) => (
                <li key={`env-${idx}`}>{item}</li>
              ))}
            </ul>
            <h3>Social</h3>
            <ul>
              {(analysis.summary?.social || []).map((item, idx) => (
                <li key={`soc-${idx}`}>{item}</li>
              ))}
            </ul>
            <h3>Governance</h3>
            <ul>
              {(analysis.summary?.governance || []).map((item, idx) => (
                <li key={`gov-${idx}`}>{item}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {summaryMetrics && (
        <section className="card">
          <h2>Summary Quality</h2>
          <div className="metrics-grid">
            {Object.entries(summaryMetrics).map(([key, value]) => (
              <div key={key} className="metric-tile">
                <span>{key.replace(/_/g, ' ')}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2>Industry Peers</h2>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <label htmlFor="industry-select" style={{ margin: 0 }}>Industry</label>
            <select
              id="industry-select"
              value={industryFilter}
              onChange={(e) => setIndustryFilter(e.target.value)}
            >
              <option value="">All</option>
              {peerData.industries.map((industry) => (
                <option key={industry} value={industry}>
                  {industry}
                </option>
              ))}
            </select>
            <button onClick={fetchPeers}>Refresh</button>
          </div>
        </div>
        <table className="peer-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Company</th>
              <th>ESG Risk</th>
            </tr>
          </thead>
          <tbody>
            {peers.map((peer) => (
              <tr key={peer.Symbol}>
                <td>{peer.Symbol}</td>
                <td>{peer.Name}</td>
                <td>{peer['Total ESG Risk score'] ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
