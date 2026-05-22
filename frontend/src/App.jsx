import { useState, useEffect } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  BarChart, Bar, CartesianGrid, Legend, LineChart, Line, ReferenceLine 
} from 'recharts';
import { 
  Activity, TrendingUp, TrendingDown, Target, Shield, 
  BookOpen, Layers, Zap, Sliders, ChevronRight, Settings, RefreshCw
} from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:8000/api';

const REGIME_COLORS = {
  "Expansion": "#3fb950",
  "Peak": "#d29922",
  "Contraction": "#f85149",
  "Recovery": "#58a6ff"
};

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState({
    overview: null,
    predictions: null,
    sentiment: null,
    historical: null,
    regimes: null,
    risk: null,
    backtest: null,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [overviewRes, predsRes, sentimentRes, histRes, regimeRes, riskRes, btRes] = await Promise.all([
          fetch(`${API_URL}/overview`),
          fetch(`${API_URL}/predictions`),
          fetch(`${API_URL}/sentiment`),
          fetch(`${API_URL}/historical`),
          fetch(`${API_URL}/regimes`),
          fetch(`${API_URL}/risk`),
          fetch(`${API_URL}/backtest`),
        ]);

        const overview = await overviewRes.json();
        const predictions = await predsRes.json();
        const sentiment = await sentimentRes.json();
        const historical = await histRes.json();
        const regimes = await regimeRes.json();
        const risk = await riskRes.json();
        const backtest = await btRes.json();

        setData({ overview, predictions, sentiment, historical, regimes, risk, backtest });
        setLoading(false);
      } catch (err) {
        console.error("Error fetching data:", err);
        setLoading(false);
      }
    }
    fetchData();

    // WebSocket auto-refresh: reconnect on data updates from backend
    let ws;
    function connectWS() {
      try {
        ws = new WebSocket('ws://localhost:8000/ws/updates');
        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'update') {
              console.log('[WS] Data update detected — refreshing...');
              fetchData();
            }
          } catch {
            return;
          }
        };
        ws.onclose = () => setTimeout(connectWS, 5000);
        ws.onerror = () => ws.close();
      } catch {
        return;
      }
    }
    connectWS();
    return () => { if (ws) ws.close(); };
  }, []);


  if (loading) return <div className="loader">Initializing AI Core...</div>;

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <Activity size={24} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'text-bottom' }} />
          Macro.AI
        </div>
        <nav className="nav-menu">
          <div className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
            <Activity size={18} /> Platform Overview
          </div>
          <div className={`nav-item ${activeTab === 'market' ? 'active' : ''}`} onClick={() => setActiveTab('market')}>
            <Sliders size={18} /> Market Pulse
          </div>
          <div className={`nav-item ${activeTab === 'regimes' ? 'active' : ''}`} onClick={() => setActiveTab('regimes')}>
            <Layers size={18} /> Economic Environments
          </div>
          <div className={`nav-item ${activeTab === 'predictions' ? 'active' : ''}`} onClick={() => setActiveTab('predictions')}>
            <Target size={18} /> Future Predictions
          </div>
          <div className={`nav-item ${activeTab === 'backtest' ? 'active' : ''}`} onClick={() => setActiveTab('backtest')}>
            <TrendingUp size={18} /> AI Strategy Performance
          </div>
          <div className={`nav-item ${activeTab === 'risk' ? 'active' : ''}`} onClick={() => setActiveTab('risk')}>
            <Shield size={18} /> Risk & Safety
          </div>
          <div className={`nav-item ${activeTab === 'tuning' ? 'active' : ''}`} onClick={() => setActiveTab('tuning')}>
            <Settings size={18} /> Model Tuning
          </div>
          <div className={`nav-item ${activeTab === 'simulator' ? 'active' : ''}`} onClick={() => setActiveTab('simulator')}>
            <Zap size={18} /> Scenario Tester
          </div>
          <div className={`nav-item ${activeTab === 'sentiment' ? 'active' : ''}`} onClick={() => setActiveTab('sentiment')}>
            <BookOpen size={18} /> Live News Sentiment
          </div>
        </nav>
        <div className="sidebar-footer">
          Institutional v2.1
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {activeTab === 'overview' && <PlatformOverviewTab />}
        {activeTab === 'market' && <MarketPulseTab data={data.overview} historical={data.historical} regimes={data.regimes} />}
        {activeTab === 'regimes' && <EnvironmentsTab regimes={data.regimes} historical={data.historical} />}
        {activeTab === 'predictions' && <PredictionsTab predictions={data.predictions} />}
        {activeTab === 'backtest' && <BacktestTab backtest={data.backtest} />}
        {activeTab === 'risk' && <RiskTab risk={data.risk} />}
        {activeTab === 'tuning' && <ModelTuningTab />}
        {activeTab === 'simulator' && <SimulatorTab historical={data.historical} />}
        {activeTab === 'sentiment' && <SentimentTab sentiment={data.sentiment} />}
      </main>
    </div>
  );
}

function PlatformOverviewTab() {
  return (
    <div className="animate-fade-in text-container">
      <header className="page-header">
        <h1 className="page-title">Platform Overview</h1>
        <p className="page-subtitle">Interactive sector allocation driven by macroeconomic machine learning.</p>
      </header>
      
      <div className="glass-card" style={{ padding: '2.5rem' }}>
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--accent-primary)', fontSize: '1.5rem' }}>System Capabilities</h2>
        <p style={{ marginBottom: '1.5rem', fontSize: '1.05rem', lineHeight: '1.6', color: 'var(--text-main)' }}>
          This platform processes over 15 years of macroeconomic data alongside sectoral equity indices. By combining statistical causality tests, unsupervised economic regime clustering, supervised XGBoost forecasting, and historical backtests, the platform provides institutional-grade sector intelligence.
        </p>

        <div className="grid-3" style={{ marginTop: '2.5rem', gap: '2rem' }}>
          <div className="feature-block">
            <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.1rem' }}>1. Macro & Market Pulse</h3>
            <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.5' }}>
              Track real-time indicators, interest rates, commodities, and benchmark indices with date-filterable historical charts.
            </p>
          </div>
          <div className="feature-block">
            <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.1rem' }}>2. Economic Environments</h3>
            <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.5' }}>
              Explore how K-Means clustering classifies economic regimes and review sector rotation returns across environments.
            </p>
          </div>
          <div className="feature-block">
            <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.1rem' }}>3. Future Predictions</h3>
            <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.5' }}>
              Inspect next-month directional predictions generated by XGBoost models, complete with confidence scores and top features.
            </p>
          </div>
        </div>

        <div className="grid-3" style={{ marginTop: '2rem', gap: '2rem' }}>
          <div className="feature-block">
            <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.1rem' }}>4. AI Strategy Backtests</h3>
            <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.5' }}>
              Verify strategy backtests executing prediction directives compared directly against benchmark Buy & Hold statistics.
            </p>
          </div>
          <div className="feature-block">
            <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.1rem' }}>5. Risk & Safety</h3>
            <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.5' }}>
              Evaluate annualized Sharpe ratios, maximum drawdowns, and 95% Value-at-Risk across different economic regimes.
            </p>
          </div>
          <div className="feature-block">
            <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.1rem' }}>6. Macro Shock Simulator</h3>
            <p className="text-muted" style={{ fontSize: '0.9rem', lineHeight: '1.5' }}>
              Adjust rates, oil, DXY, and VIX in real time to simulate shocks and instantly visualize the predicted impact.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function MarketPulseTab({ data, historical }) {
  const [selectedCol, setSelectedCol] = useState('Nifty_50');

  if (!data || !data.metrics || !historical || !historical.data) {
    return <div>No data available</div>;
  }

  // Get available columns for plotting
  const allCols = Object.keys(historical.data[0] || {}).filter(
    col => col !== 'Date' && !col.endsWith('_Return')
  );

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Market Pulse</h1>
        <p className="page-subtitle">Real-time macroeconomic indicators and asset performance.</p>
      </header>

      <div className="grid-3">
        {data.metrics.map(metric => (
          <div key={metric.id} className="glass-card">
            <div className="metric-header">
              <span className="metric-title">{metric.title}</span>
              {metric.delta > 0 ? <TrendingUp size={18} className={metric.is_higher_better ? "positive" : "negative"} /> : 
               metric.delta < 0 ? <TrendingDown size={18} className={metric.is_higher_better ? "negative" : "positive"} /> : null}
            </div>
            <div className="metric-value">
              {metric.id.includes('Rate') || metric.id.includes('Yield') 
                ? `${metric.value.toFixed(2)}%` 
                : metric.value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
            </div>
            <div className={`metric-delta ${metric.delta > 0 ? (metric.is_higher_better ? 'positive' : 'negative') : (metric.is_higher_better ? 'negative' : 'positive')}`}>
              {metric.delta > 0 ? '+' : ''}{metric.id.includes('Rate') || metric.id.includes('Yield') ? `${(metric.delta * 100).toFixed(0)} bps` : metric.delta.toFixed(2)} ({metric.pct_delta > 0 ? '+' : ''}{metric.pct_delta.toFixed(2)}%)
            </div>
            
            {/* Sparkline */}
            <div style={{ height: '60px', marginTop: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={metric.trend.map((v, i) => ({ val: v, i }))}>
                  <defs>
                    <linearGradient id={`grad-${metric.id}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={metric.delta > 0 ? (metric.is_higher_better ? '#10b981' : '#ef4444') : (metric.is_higher_better ? '#ef4444' : '#10b981')} stopOpacity={0.2}/>
                      <stop offset="95%" stopColor={metric.delta > 0 ? (metric.is_higher_better ? '#10b981' : '#ef4444') : (metric.is_higher_better ? '#ef4444' : '#10b981')} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="val" stroke={metric.delta > 0 ? (metric.is_higher_better ? '#10b981' : '#ef4444') : (metric.is_higher_better ? '#ef4444' : '#10b981')} strokeWidth={2} fillOpacity={1} fill={`url(#grad-${metric.id})`} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>

      <div className="glass-card" style={{ marginTop: '2.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Historical Trajectory</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Select Variable:</span>
            <select 
              value={selectedCol} 
              onChange={(e) => setSelectedCol(e.target.value)}
              className="select-input"
            >
              {allCols.map(col => (
                <option key={col} value={col}>{col.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ height: '350px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={historical.data}>
              <defs>
                <linearGradient id="selectedColorGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.2}/>
                  <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" vertical={false} />
              <XAxis dataKey="Date" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
              <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--card-border)', borderRadius: '8px' }}
                labelClassName="text-muted"
              />
              <Area type="monotone" dataKey={selectedCol} stroke="var(--accent-primary)" strokeWidth={2} fillOpacity={1} fill="url(#selectedColorGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function EnvironmentsTab({ regimes, historical }) {
  const [regimeMode, setRegimeMode] = useState('kmeans'); // 'kmeans' | 'hmm'
  const [hmmTimeline, setHmmTimeline] = useState(null);
  const [hmmLoading, setHmmLoading] = useState(false);

  if (!regimes || !regimes.timeline || !regimes.rotation || !historical || !historical.data) {
    return <div>No data available</div>;
  }

  const { sectors, raw_sectors, environments, matrix } = regimes.rotation;

  const handleHmmToggle = async (mode) => {
    setRegimeMode(mode);
    if (mode === 'hmm' && !hmmTimeline) {
      setHmmLoading(true);
      try {
        const res = await fetch(`${API_URL}/regimes/hmm`);
        const json = await res.json();
        setHmmTimeline(json.timeline || []);
      } catch {
        setHmmTimeline([]);
      }
      setHmmLoading(false);
    }
  };

  // Build lookup mapping dates to regimes (K-Means or HMM)
  const activeTimeline = regimeMode === 'hmm' ? (hmmTimeline || []) : regimes.timeline;
  const regimeMap = {};
  activeTimeline.forEach(item => {
    regimeMap[item.Date] = item.HMM_Regime || item.Regime;
  });

  // Merge historical Nifty 50 with regime timeline
  const timelineData = historical.data.map(item => ({
    Date: item.Date,
    Nifty_50: item.Nifty_50,
    regime: regimeMap[item.Date] || 'Unknown'
  })).filter(item => item.Nifty_50 !== null);

  // Custom Dot for coloring regime line markers
  const renderRegimeDot = (props) => {
    const { cx, cy, payload } = props;
    if (!cx || !cy) return null;
    const color = REGIME_COLORS[payload.regime];
    if (!color) return null;
    return (
      <circle key={`${payload.Date}`} cx={cx} cy={cy} r={3} fill={color} stroke="none" />
    );
  };

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Economic Environments</h1>
        <p className="page-subtitle">Unsupervised clustering and resulting sector returns.</p>
      </header>

      {/* Rotation Heatmap Matrix */}
      <div className="glass-card" style={{ marginBottom: '2.5rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>Sector Rotation Heatmap Matrix</h2>
        <p className="text-muted" style={{ fontSize: '0.9rem', marginBottom: '2rem' }}>
          Average historical annualized sector performance across mathematically defined economic states.
        </p>

        <div style={{ overflowX: 'auto' }}>
          <table className="heatmap-table">
            <thead>
              <tr>
                <th>Macro State</th>
                {sectors.map(sec => (
                  <th key={sec}>{sec.replace('Nifty ', '')}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {environments.map(env => (
                <tr key={env}>
                  <td style={{ fontWeight: 600, color: 'var(--text-main)', minWidth: '130px' }}>{env}</td>
                  {raw_sectors.map(col => {
                    const val = matrix[env]?.[col];
                    if (val === undefined || val === null) {
                      return <td key={col} className="heatmap-cell">N/A</td>;
                    }
                    const displayVal = (val * 100).toFixed(1) + '%';
                    const opacity = Math.min(Math.abs(val) * 3, 0.8);
                    const backgroundColor = val > 0 
                      ? `rgba(16, 185, 129, ${opacity})`
                      : `rgba(239, 68, 68, ${opacity})`;
                    return (
                      <td key={col} className="heatmap-cell" style={{ backgroundColor }}>
                        {displayVal}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Regime Timeline — with K-Means / HMM toggle */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.25rem' }}>Regime Timeline Mapping</h2>
            <p className="text-muted" style={{ fontSize: '0.9rem' }}>
              {regimeMode === 'kmeans'
                ? 'Historical Nifty 50 levels marked by K-Means economic clusters.'
                : 'Historical Nifty 50 levels marked by Hidden Markov Model states.'}
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* Model toggle */}
            <div style={{ display: 'flex', background: 'var(--sidebar-bg)', borderRadius: '8px', padding: '3px', border: '1px solid var(--card-border)' }}>
              {['kmeans', 'hmm'].map(mode => (
                <button
                  key={mode}
                  onClick={() => handleHmmToggle(mode)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: '6px',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    transition: 'all 0.2s',
                    background: regimeMode === mode ? 'var(--accent-primary)' : 'transparent',
                    color: regimeMode === mode ? '#fff' : 'var(--text-muted)',
                  }}
                >
                  {mode === 'kmeans' ? 'K-Means' : 'HMM'}
                </button>
              ))}
            </div>

            {/* Legend dots */}
            <div style={{ display: 'flex', gap: '12px' }}>
              {Object.entries(REGIME_COLORS).map(([reg, col]) => (
                <div key={reg} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.82rem' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '3px', backgroundColor: col }} />
                  <span>{reg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {hmmLoading ? (
          <div style={{ height: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            <RefreshCw size={20} style={{ marginRight: '8px', animation: 'spin 1s linear infinite' }} />
            Running HMM regime detection...
          </div>
        ) : (
          <div style={{ height: '350px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" vertical={false} />
                <XAxis dataKey="Date" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--card-border)', borderRadius: '8px' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const row = payload[0].payload;
                      return (
                        <div style={{ backgroundColor: 'var(--card-bg)', border: '1px solid var(--card-border)', padding: '10px', borderRadius: '8px' }}>
                          <p className="text-muted" style={{ fontSize: '0.8rem', marginBottom: '4px' }}>{row.Date}</p>
                          <p style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-main)' }}>Nifty 50: {row.Nifty_50?.toLocaleString(undefined, { maximumFractionDigits: 1 })}</p>
                          <p style={{ fontWeight: 600, color: REGIME_COLORS[row.regime] || '#fff', fontSize: '0.85rem', marginTop: '4px' }}>Regime: {row.regime}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line type="monotone" dataKey="Nifty_50" stroke="#798296" strokeWidth={1.5} dot={renderRegimeDot} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

function PredictionsTab({ predictions }) {
  const [selectedSec, setSelectedSec] = useState('');

  if (!predictions || !predictions.predictions) {
    return <div>No predictions available</div>;
  }

  const selectedValue = selectedSec || predictions.predictions[0]?.sector || '';
  const selectedPred = predictions.predictions.find(p => p.sector === selectedValue) || predictions.predictions[0];

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Future Predictions</h1>
        <p className="page-subtitle">Next-month directional forecasting powered by XGBoost machine learning.</p>
      </header>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '2.5rem' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>Select Target Sector:</span>
        <select 
          value={selectedValue} 
          onChange={(e) => setSelectedSec(e.target.value)}
          className="select-input"
        >
          {predictions.predictions.map(p => (
            <option key={p.sector} value={p.sector}>{p.sector}</option>
          ))}
        </select>
      </div>

      {selectedPred && (
        <div className="grid-2">
          {/* Main Directional Outcome Card */}
          <div className={`glass-card prediction-card ${selectedPred.prediction === 1 ? 'up' : 'down'}`} style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '2.5rem' }}>
            <div>
              <div className="metric-title" style={{ marginBottom: '1.5rem' }}>Next Month Forecast</div>
              <h2 style={{ fontSize: '2.25rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '0.5rem' }}>{selectedPred.sector}</h2>
              <p className="text-muted" style={{ fontSize: '0.9rem' }}>Directional indicator mapping macro momentum.</p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '15px', margin: '2rem 0' }}>
              {selectedPred.prediction === 1 ? 
                <TrendingUp size={44} className="positive" /> : 
                <TrendingDown size={44} className="negative" />
              }
              <div>
                <div style={{ fontSize: '2rem', fontWeight: 800, color: selectedPred.prediction === 1 ? 'var(--accent-green)' : 'var(--accent-red)', lineHeight: 1.1 }}>
                  {selectedPred.prediction === 1 ? 'BULLISH' : 'BEARISH'}
                </div>
                <div className="text-muted" style={{ fontSize: '0.95rem', marginTop: '4px' }}>
                  Model Confidence: <strong>{(selectedPred.confidence * 100).toFixed(1)}%</strong>
                </div>
              </div>
            </div>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: '1.5' }}>
              Predictions represent model classifications based on the latest feature matrices. Unscheduled fundamental shocks or central bank directives can alter real-world returns.
            </p>
          </div>

          {/* Feature Importances Card */}
          <div className="glass-card">
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '1.5rem' }}>Top Predictive Model Drivers</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart 
                  data={selectedPred.top_drivers.map(driver => ({
                    feature: driver.Feature.replace(/_/g, ' '),
                    importance: driver.Importance * 100
                  }))}
                  layout="vertical"
                  margin={{ left: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" horizontal={false} />
                  <XAxis type="number" stroke="var(--text-muted)" fontSize={11} unit="%" />
                  <YAxis type="category" dataKey="feature" stroke="var(--text-muted)" fontSize={10} width={120} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--card-border)', borderRadius: '8px' }}
                    labelClassName="text-muted"
                  />
                  <Bar dataKey="importance" fill="var(--accent-primary)" radius={[0, 4, 4, 0]} barSize={18} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function BacktestTab({ backtest }) {
  const [selectedSec, setSelectedSec] = useState('');

  if (!backtest || !backtest.results) {
    return <div>No backtest results available</div>;
  }

  const rawSectors = Object.keys(backtest.results);
  const selectedValue = selectedSec || rawSectors[0] || '';
  const selectedStats = backtest.results[selectedValue] || backtest.results[rawSectors[0]];

  // Prepare data for comparisons
  const compareData = rawSectors.map(key => {
    const s = backtest.results[key];
    return {
      name: key.replace(/_/g, ' '),
      "AI Strategy": s.strategy_return * 100,
      "Buy & Hold": s.buyhold_return * 100
    };
  });

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Strategy Performance</h1>
        <p className="page-subtitle">Backtest simulation of AI-directed allocation versus a Buy & Hold benchmark.</p>
      </header>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '2.5rem' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>Select Sector:</span>
        <select 
          value={selectedValue} 
          onChange={(e) => setSelectedSec(e.target.value)}
          className="select-input"
        >
          {rawSectors.map(key => (
            <option key={key} value={key}>{key.replace(/_/g, ' ')}</option>
          ))}
        </select>
      </div>

      {selectedStats && (
        <>
          <div className="grid-4" style={{ marginBottom: '2.5rem' }}>
            <div className="glass-card">
              <span className="metric-title">AI Strategy Return</span>
              <div className="metric-value" style={{ color: selectedStats.strategy_return > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {selectedStats.strategy_return > 0 ? '+' : ''}{(selectedStats.strategy_return * 100).toFixed(1)}%
              </div>
            </div>
            <div className="glass-card">
              <span className="metric-title">Buy & Hold Return</span>
              <div className="metric-value" style={{ color: selectedStats.buyhold_return > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                {selectedStats.buyhold_return > 0 ? '+' : ''}{(selectedStats.buyhold_return * 100).toFixed(1)}%
              </div>
            </div>
            <div className="glass-card">
              <span className="metric-title">Prediction Accuracy</span>
              <div className="metric-value">
                {selectedStats.win_rate.toFixed(1)}%
              </div>
            </div>
            <div className="glass-card">
              <span className="metric-title">Max Drawdown</span>
              <div className="metric-value" style={{ color: 'var(--accent-red)' }}>
                {(selectedStats.max_drawdown * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Cross Sector Bar Chart */}
          <div className="glass-card">
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '2rem' }}>Cumulative Return Comparison</h3>
            <div style={{ height: '350px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={compareData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" vertical={false} />
                  <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                  <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} unit="%" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--card-border)', borderRadius: '8px' }}
                    labelClassName="text-muted"
                  />
                  <Legend />
                  <Bar dataKey="AI Strategy" fill="var(--accent-green)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Buy & Hold" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function RiskTab({ risk }) {
  const [selectedRegime, setSelectedRegime] = useState('All Regimes');

  if (!risk || !risk.metrics) {
    return <div>No risk metrics data available</div>;
  }

  const regimes = ['All Regimes', 'Peak', 'Recovery', 'Expansion', 'Contraction'];
  const filteredMetrics = risk.metrics.filter(m => m.regime === selectedRegime);

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Risk & Safety</h1>
        <p className="page-subtitle">Historical risk statistics segmented across different economic regimes.</p>
      </header>

      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '2.5rem' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>Economic Environment Filter:</span>
        <select 
          value={selectedRegime} 
          onChange={(e) => setSelectedRegime(e.target.value)}
          className="select-input"
        >
          {regimes.map(reg => (
            <option key={reg} value={reg}>{reg}</option>
          ))}
        </select>
      </div>

      <div className="glass-card">
        <div style={{ overflowX: 'auto' }}>
          <table className="risk-table">
            <thead>
              <tr>
                <th>Sector Name</th>
                <th>Sharpe Ratio (Risk-Adjusted)</th>
                <th>Max Loss % (Drawdown)</th>
                <th>95% Value at Risk (VaR)</th>
                <th>Historical Months</th>
              </tr>
            </thead>
            <tbody>
              {filteredMetrics.map((row, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600, color: 'var(--text-main)' }}>{row.sector}</td>
                  <td>{row.sharpe_ratio !== null ? row.sharpe_ratio.toFixed(3) : 'N/A'}</td>
                  <td className="negative">{row.max_drawdown !== null ? `${(row.max_drawdown * 100).toFixed(2)}%` : 'N/A'}</td>
                  <td className="negative">{row.var_95 !== null ? `${(row.var_95 * 100).toFixed(2)}%` : 'N/A'}</td>
                  <td>{row.months}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ModelTuningTab() {
  const sectorOptions = [
    'Nifty_50_Return',
    'Nifty_Bank_Return',
    'Nifty_IT_Return',
    'Nifty_Pharma_Return',
    'Nifty_Auto_Return',
    'Nifty_FMCG_Return',
    'Nifty_Metal_Return',
    'Nifty_Realty_Return',
    'Nifty_Energy_Return',
    'Nifty_Infra_Return',
    'Nifty_PSE_Return',
    'Nifty_Media_Return',
  ];

  const [selectedSector, setSelectedSector] = useState(sectorOptions[0]);
  const [tuning, setTuning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleTune = async () => {
    setTuning(true);
    setError('');
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/tune/${encodeURIComponent(selectedSector)}`);
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Tuning request failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message || 'Unable to run model tuning');
    } finally {
      setTuning(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Model Tuning</h1>
        <p className="page-subtitle">Run Optuna hyperparameter search for a selected sector model.</p>
      </header>

      <div className="grid-2">
        <div className="glass-card">
          <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '1.5rem' }}>
            Tuning Control
          </h3>

          <div style={{ marginBottom: '1.5rem' }}>
            <label className="metric-title" htmlFor="tuning-sector" style={{ display: 'block', marginBottom: '0.75rem' }}>
              Target Sector
            </label>
            <select
              id="tuning-sector"
              value={selectedSector}
              onChange={(e) => setSelectedSector(e.target.value)}
              className="select-input"
              style={{ width: '100%' }}
              disabled={tuning}
            >
              {sectorOptions.map(sector => (
                <option key={sector} value={sector}>
                  {sector.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleTune}
            disabled={tuning}
            className="sim-btn"
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            {tuning && <RefreshCw size={18} style={{ marginRight: '8px', animation: 'spin 1s linear infinite' }} />}
            {tuning ? 'Running Optuna Search...' : 'Start Tuning'}
          </button>

          <p className="text-muted" style={{ marginTop: '1.5rem', fontSize: '0.85rem', lineHeight: 1.6 }}>
            Tuning may take a while because the backend runs multiple validation trials before returning the best parameter set.
          </p>
        </div>

        <div className="glass-card">
          <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '1.5rem' }}>
            Latest Tuning Result
          </h3>

          {error && (
            <div style={{ color: 'var(--accent-red)', lineHeight: 1.5 }}>
              {error}
            </div>
          )}

          {!error && !result && (
            <div style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Select a target sector and start tuning to view the best validation score and model parameters.
            </div>
          )}

          {result && (
            <>
              <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                <div>
                  <span className="metric-title">Sector</span>
                  <div style={{ marginTop: '0.4rem', fontWeight: 700 }}>
                    {(result.sector || selectedSector).replace(/_/g, ' ')}
                  </div>
                </div>
                <div>
                  <span className="metric-title">Best F1 Score</span>
                  <div style={{ marginTop: '0.4rem', fontWeight: 700, color: 'var(--accent-green)' }}>
                    {typeof result.best_score === 'number' ? result.best_score.toFixed(4) : 'N/A'}
                  </div>
                </div>
              </div>

              <pre style={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid var(--card-border)',
                borderRadius: '12px',
                padding: '1rem',
                color: 'var(--text-main)',
                fontSize: '0.85rem',
                lineHeight: 1.5,
              }}>
                {JSON.stringify(result.best_params || result, null, 2)}
              </pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function SimulatorTab({ historical }) {
  const [inputs, setInputs] = useState(() => {
    const latest = historical?.data?.[historical.data.length - 1] || {};
    return {
      US_Fed_Funds_Rate: latest.US_Fed_Funds_Rate || 4.5,
      Brent_Crude: latest.Brent_Crude || 75.0,
      DXY: latest.DXY || 100.0,
      India_VIX: latest.India_VIX || 16.0,
    };
  });
  const [results, setResults] = useState(null);
  const [simulating, setSimulating] = useState(false);

  const handleSimulate = async () => {
    setSimulating(true);
    try {
      const res = await fetch(`${API_URL}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputs)
      });
      const data = await res.json();
      if (data.results) {
        // Sort results by probability ascending for horizontal bar chart
        const sorted = [...data.results].sort((a, b) => a.probability - b.probability);
        setResults(sorted);
      }
    } catch (err) {
      console.error("Simulation failed:", err);
    }
    setSimulating(false);
  };

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Scenario Tester</h1>
        <p className="page-subtitle">Stress-test indicators to simulate shocks and view real-time adjustments.</p>
      </header>

      <div className="grid-2">
        {/* Controls Card */}
        <div className="glass-card" style={{ padding: '2rem' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '2rem', color: 'var(--text-main)' }}>Adjust Macro Variables</h3>
          
          <div className="slider-group">
            <div className="slider-label">
              <span>US Interest Rate</span>
              <strong>{inputs.US_Fed_Funds_Rate.toFixed(2)}%</strong>
            </div>
            <input 
              type="range" 
              min="0.0" 
              max="10.0" 
              step="0.25"
              value={inputs.US_Fed_Funds_Rate} 
              onChange={(e) => setInputs({...inputs, US_Fed_Funds_Rate: parseFloat(e.target.value)})}
              className="slider-input"
            />
          </div>

          <div className="slider-group">
            <div className="slider-label">
              <span>Oil Price (Brent Crude)</span>
              <strong>${inputs.Brent_Crude.toFixed(1)}</strong>
            </div>
            <input 
              type="range" 
              min="20.0" 
              max="150.0" 
              step="5.0"
              value={inputs.Brent_Crude} 
              onChange={(e) => setInputs({...inputs, Brent_Crude: parseFloat(e.target.value)})}
              className="slider-input"
            />
          </div>

          <div className="slider-group">
            <div className="slider-label">
              <span>US Dollar Strength (DXY)</span>
              <strong>{inputs.DXY.toFixed(1)}</strong>
            </div>
            <input 
              type="range" 
              min="70.0" 
              max="130.0" 
              step="1.0"
              value={inputs.DXY} 
              onChange={(e) => setInputs({...inputs, DXY: parseFloat(e.target.value)})}
              className="slider-input"
            />
          </div>

          <div className="slider-group">
            <div className="slider-label">
              <span>Market Fear (India VIX)</span>
              <strong>{inputs.India_VIX.toFixed(1)}</strong>
            </div>
            <input 
              type="range" 
              min="10.0" 
              max="80.0" 
              step="1.0"
              value={inputs.India_VIX} 
              onChange={(e) => setInputs({...inputs, India_VIX: parseFloat(e.target.value)})}
              className="slider-input"
            />
          </div>

          <button 
            onClick={handleSimulate} 
            disabled={simulating}
            className="sim-btn"
          >
            {simulating ? 'Evaluating Shock...' : 'Run Simulation'}
          </button>
        </div>

        {/* Results Card */}
        <div className="glass-card" style={{ minHeight: '400px' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '2rem', color: 'var(--text-main)' }}>AI Output: Bullish Class Probabilities</h3>
          {results ? (
            <div style={{ height: '350px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={results} layout="vertical" margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" horizontal={false} />
                  <XAxis type="number" domain={[0, 100]} stroke="var(--text-muted)" fontSize={11} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="sector" stroke="var(--text-muted)" fontSize={10} width={90} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--card-border)', borderRadius: '8px' }}
                    labelClassName="text-muted"
                    formatter={(val) => [`${(val).toFixed(1)}%`, 'Bullish Probability']}
                  />
                  <ReferenceLine x={50} stroke="var(--text-muted)" strokeDasharray="4 4" />
                  <Bar 
                    dataKey="probability" 
                    fill="var(--accent-primary)" 
                    radius={[0, 4, 4, 0]}
                    barSize={15}
                    // Transform raw 0-1 prob to 0-100 percentage
                    data={results.map(r => ({...r, probability: r.probability * 100}))}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '80%', color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
              <Activity size={48} style={{ marginBottom: '1rem', opacity: 0.3 }} />
              <p>Adjust inputs and click "Run Simulation" to observe how sector forecast dynamics shift in response to the economic shock.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SentimentTab({ sentiment }) {
  if (!sentiment) return <div>No sentiment data available</div>;

  return (
    <div className="animate-fade-in">
      <header className="page-header">
        <h1 className="page-title">Live News Sentiment</h1>
        <p className="page-subtitle">Real-time VADER news sentiment scoring.</p>
      </header>

      <div className="grid-2">
        {/* Aggregate Score Card */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', padding: '3rem' }}>
          <div className="metric-title" style={{ marginBottom: '1.5rem' }}>Aggregate Market Mood</div>
          <div style={{ fontSize: '5rem', fontWeight: 800, lineHeight: 1, color: sentiment.score > 0.05 ? 'var(--accent-green)' : (sentiment.score < -0.05 ? 'var(--accent-red)' : 'var(--accent-purple)') }}>
            {sentiment.score > 0 ? '+' : ''}{sentiment.score.toFixed(3)}
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 600, marginTop: '1.5rem', color: 'var(--text-main)' }}>{sentiment.label}</div>
          <p style={{ color: 'var(--text-muted)', marginTop: '2.5rem', fontSize: '0.9rem', lineHeight: '1.6' }}>
            Computed by analyzing {sentiment.articles.length} live headlines with Natural Language Processing. Scores above +0.05 are Bullish, below -0.05 Bearish.
          </p>
        </div>

        {/* Headlines Card */}
        <div className="glass-card" style={{ overflowY: 'auto', maxHeight: '550px', padding: '2rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '1.5rem' }}>Latest News Feed</h3>
          {sentiment.articles.map((art, idx) => (
            <a 
              key={idx} 
              href={art.link} 
              target="_blank" 
              rel="noopener noreferrer" 
              className="news-item-link"
              style={{ display: 'block', textDecoration: 'none', color: 'inherit' }}
            >
              <div className="news-item">
                <div className="news-header-row">
                  <span className="news-title">{art.title}</span>
                  <ChevronRight size={14} className="news-arrow" />
                </div>
                <div className="news-meta">
                  <span style={{ 
                    color: art.label === 'Positive' ? 'var(--accent-green)' : (art.label === 'Negative' ? 'var(--accent-red)' : 'var(--accent-purple)'),
                    fontWeight: 600 
                  }}>
                    {art.label} ({art.compound > 0 ? '+' : ''}{art.compound.toFixed(2)})
                  </span>
                  <span style={{ margin: '0 8px' }}>|</span>
                  {art.publisher}
                  <span style={{ margin: '0 8px' }}>|</span>
                  {art.published}
                </div>
              </div>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
