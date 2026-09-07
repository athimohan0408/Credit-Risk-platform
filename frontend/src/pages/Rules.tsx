import { useEffect, useState } from 'react';
import api from '../api';

type Rule = {
  rank: number;
  feature: string;
  description: string;
  mean_abs_shap: number;
  value_shap_corr: number;
  type: 'numeric' | 'categorical';
  condition: string;
  direction: string;
  segment_size: number;
  segment_share_pct: number;
  segment_default_rate_pct: number;
  lift_vs_baseline: number;
  rule_text: string;
  categories?: { value: string; default_rate_pct: number; n: number }[];
};

const liftColor = (lift: number) =>
  lift >= 1.75 ? '#e74c3c' : lift >= 1.3 ? '#f39c12' : '#3498db';

const Rules = () => {
  const [data, setData] = useState<any>(null);
  const [showChecks, setShowChecks] = useState(false);
  const [showRejected, setShowRejected] = useState(false);

  useEffect(() => {
    api.get('/explainability/rules').then(res => setData(res.data)).catch(console.error);
  }, []);

  if (!data) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
      <p style={{ color: '#aaa' }}>Loading Business Rules...</p>
    </div>
  );

  const rules: Rule[] = data.rules || [];
  const baseline: number = data.baseline_default_rate_pct ?? 0;
  const checks: string[] = data.sanity_checks || [];
  const rejected: { feature: string; reason: string }[] = data.rejected || [];
  const failures = checks.filter(c => c.startsWith('[FAIL]')).length;

  const sorted = [...rules].sort((a, b) => b.lift_vs_baseline - a.lift_vs_baseline);
  const strongest = sorted[0];

  return (
    <div>
      <h1 style={{ marginBottom: '4px' }}>📋 Business Rules</h1>
      <p style={{ color: '#aaa', marginTop: 0 }}>
        Derived from SHAP values, then validated against observed default rates. Each rule
        reports the real risk lift of the segment it flags.
      </p>

      {/* Headline stats */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
        <div className="glass-card" style={{ flex: 1, minWidth: '140px', padding: '10px 16px' }}>
          <span className="metric-label">Portfolio Baseline</span>
          <div style={{ color: '#a18cd1', fontWeight: 700, fontSize: '1.3rem' }}>{baseline}%</div>
        </div>
        <div className="glass-card" style={{ flex: 1, minWidth: '140px', padding: '10px 16px' }}>
          <span className="metric-label">Accepted Rules</span>
          <div style={{ color: '#2ecc71', fontWeight: 700, fontSize: '1.3rem' }}>{rules.length}</div>
        </div>
        <div className="glass-card" style={{ flex: 1, minWidth: '140px', padding: '10px 16px' }}>
          <span className="metric-label">Strongest Lift</span>
          <div style={{ color: '#e74c3c', fontWeight: 700, fontSize: '1.3rem' }}>
            {strongest ? `${strongest.lift_vs_baseline}x` : '—'}
          </div>
        </div>
        <div className="glass-card" style={{ flex: 1, minWidth: '140px', padding: '10px 16px' }}>
          <span className="metric-label">Sanity Checks</span>
          <div style={{ color: failures ? '#e74c3c' : '#2ecc71', fontWeight: 700, fontSize: '1.3rem' }}>
            {checks.length - failures}/{checks.length}
          </div>
        </div>
      </div>

      {/* Rules */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '28px' }}>
        {sorted.map(r => {
          const color = liftColor(r.lift_vs_baseline);
          return (
            <div
              key={r.feature}
              style={{
                background: `${color}12`,
                borderLeft: `4px solid ${color}`,
                borderRadius: '10px',
                padding: '14px 18px',
                transition: 'transform 0.15s',
              }}
              onMouseEnter={e => (e.currentTarget.style.transform = 'translateX(4px)')}
              onMouseLeave={e => (e.currentTarget.style.transform = 'translateX(0)')}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '10px' }}>
                <div style={{ flex: 1, minWidth: '260px' }}>
                  <span style={{ color, fontWeight: 800, fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
                    {r.description} · {r.type}
                  </span>
                  <div style={{ marginTop: '8px', fontSize: '0.98rem', fontWeight: 600, color: '#e0e0ff' }}>
                    IF {r.condition}
                  </div>
                  <div style={{ marginTop: '4px', fontSize: '0.85rem', color: '#9a9ab0' }}>
                    → flag for enhanced underwriting · {r.direction}
                  </div>

                  {r.categories && r.categories.length > 0 && (
                    <div style={{ marginTop: '8px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                      {r.categories.map(c => (
                        <span key={c.value} style={{
                          background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)',
                          borderRadius: '10px', padding: '2px 9px', fontSize: '0.74rem', color: '#bbb',
                        }}>
                          {c.value}: {c.default_rate_pct}% (n={c.n.toLocaleString()})
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '5px', minWidth: '150px' }}>
                  <span style={{
                    background: `${color}22`, color, border: `1px solid ${color}55`,
                    borderRadius: '12px', padding: '3px 12px', fontSize: '0.85rem', fontWeight: 800, whiteSpace: 'nowrap',
                  }}>
                    {r.lift_vs_baseline}x risk
                  </span>
                  <span style={{ color: '#888', fontSize: '0.74rem' }}>
                    <strong style={{ color: '#ddd' }}>{r.segment_default_rate_pct}%</strong> default vs {baseline}%
                  </span>
                  <span style={{ color: '#888', fontSize: '0.74rem' }}>
                    covers <strong style={{ color: '#ddd' }}>{r.segment_share_pct}%</strong> of applicants
                  </span>
                  <span style={{ color: '#888', fontSize: '0.74rem' }}>
                    SHAP <strong style={{ color: '#a18cd1' }}>{r.mean_abs_shap.toFixed(4)}</strong> · corr{' '}
                    <strong style={{ color: '#a18cd1' }}>{r.value_shap_corr > 0 ? '+' : ''}{r.value_shap_corr}</strong>
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Sanity checks */}
      {checks.length > 0 && (
        <div className="glass-panel" style={{ marginBottom: '16px' }}>
          <div
            style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            onClick={() => setShowChecks(!showChecks)}
          >
            <h3 style={{ margin: 0 }}>🔬 Directional Sanity Checks</h3>
            <span style={{ color: '#a18cd1', fontSize: '0.85rem' }}>
              {showChecks ? '▲ Hide' : '▼ Show'} · {checks.length - failures}/{checks.length} passed
            </span>
          </div>
          {showChecks && (
            <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {checks.map((w, i) => {
                const isOk = w.startsWith('[OK]');
                const color = isOk ? '#2ecc71' : w.startsWith('[FAIL]') ? '#e74c3c' : '#3498db';
                return (
                  <div key={i} style={{
                    background: `${color}11`, borderLeft: `3px solid ${color}`,
                    borderRadius: '6px', padding: '8px 12px', fontSize: '0.82rem', color: '#ccc',
                  }}>
                    {w}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Rejected candidates */}
      {rejected.length > 0 && (
        <div className="glass-panel">
          <div
            style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            onClick={() => setShowRejected(!showRejected)}
          >
            <h3 style={{ margin: 0 }}>🚫 Rejected Candidates</h3>
            <span style={{ color: '#a18cd1', fontSize: '0.85rem' }}>
              {showRejected ? '▲ Hide' : '▼ Show'} {rejected.length}
            </span>
          </div>
          {showRejected && (
            <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <p style={{ color: '#888', fontSize: '0.8rem', margin: '0 0 4px' }}>
                High-SHAP features whose flagged segment did not default more than the baseline.
                They are reported rather than emitted as rules.
              </p>
              {rejected.map((r, i) => (
                <div key={i} style={{
                  background: 'rgba(243,156,18,0.08)', borderLeft: '3px solid #f39c12',
                  borderRadius: '6px', padding: '8px 12px', fontSize: '0.82rem', color: '#ccc',
                }}>
                  <strong style={{ color: '#f39c12' }}>{r.feature}</strong> — {r.reason}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Rules;
