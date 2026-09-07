import { useEffect, useState } from 'react';
import api from '../api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const GRADIENT_COLORS = [
  '#e74c3c','#e67e22','#f39c12','#f1c40f','#2ecc71',
  '#1abc9c','#3498db','#9b59b6','#a18cd1','#fbc2eb',
  '#fd79a8','#6c5ce7','#00b894','#fdcb6e','#e17055',
];

const Explainability = () => {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    api.get('/explainability/feature_importance').then(res => setData(res.data)).catch(console.error);
  }, []);

  if (!data) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'60vh' }}>
      <p style={{ color:'#aaa' }}>Loading Explainability...</p>
    </div>
  );

  const chartData = Object.entries(data.top_features || {})
    .map(([name, value]) => ({ name, value }))
    .sort((a: any, b: any) => b.value - a.value)
    .slice(0, 15);

  const foldAucs: number[] = data.fold_aucs || [];
  const cm = data.confusion_matrix || {};
  const op = data.operating_point || {};
  const cal = data.calibration || {};
  const hasCm = cm.TP != null;

  return (
    <div>
      <h1 style={{ marginBottom: '4px' }}>🔍 Model Explainability</h1>
      <p style={{ color: '#aaa', marginTop: 0 }}>Global feature importance from LightGBM split counts · SHAP per-prediction available on Predict page</p>

      {/* Model Performance Cards */}
      <div style={{ display: 'flex', gap: '16px', marginBottom: '24px', flexWrap: 'wrap' }}>
        {[
          { label: 'CV ROC-AUC',  value: data.cv_roc_auc  || '0.7651', sub: '5-fold average' },
          { label: 'CV PR-AUC',   value: data.cv_pr_auc   || '0.2525', sub: 'Handles imbalance' },
          { label: 'Features',    value: data.n_features   || '138',    sub: 'After engineering' },
          { label: 'Train Size',  value: data.n_samples ? Number(data.n_samples).toLocaleString() : '307,511', sub: 'Applicants' },
        ].map(({ label, value, sub }) => (
          <div key={label} className="glass-card" style={{ flex: 1, minWidth: '140px' }}>
            <div className="metric-label">{label}</div>
            <div className="metric-value">{value}</div>
            <div style={{ color: '#888', fontSize: '0.75rem' }}>{sub}</div>
          </div>
        ))}
      </div>

      {/* Per-Fold AUCs */}
      {foldAucs.length > 0 && (
        <div className="glass-panel" style={{ marginBottom: '24px' }}>
          <h3>Per-Fold ROC-AUC (Tight spread = no overfitting)</h3>
          <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
            {foldAucs.map((auc: number, i: number) => (
              <div key={i} style={{
                flex: 1, background: 'rgba(161,140,209,0.12)', borderRadius: '10px',
                padding: '14px', textAlign: 'center', border: '1px solid rgba(161,140,209,0.25)'
              }}>
                <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '4px' }}>Fold {i+1}</div>
                <div style={{ color: '#a18cd1', fontWeight: 700, fontSize: '1.2rem' }}>{auc.toFixed(4)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Confusion matrix + calibration */}
      {hasCm && (
        <div className="glass-panel" style={{ marginBottom: '24px' }}>
          <h3 style={{ marginBottom: '4px' }}>Out-of-Fold Performance</h3>
          <p style={{ color: '#888', fontSize: '0.82rem', marginTop: 0, marginBottom: '16px' }}>
            Measured on cross-validated predictions at the chosen review threshold
            of {(cm.threshold * 100).toFixed(0)}% calibrated probability — never on data the model trained on.
          </p>

          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            {/* Matrix */}
            <div style={{ flex: '1 1 320px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '8px', color: '#888', fontWeight: 500 }}></th>
                    <th style={{ padding: '8px', color: '#a18cd1' }}>Predicted OK</th>
                    <th style={{ padding: '8px', color: '#a18cd1' }}>Predicted Default</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ padding: '8px', color: '#a18cd1', fontWeight: 700 }}>Actually OK</td>
                    <td style={{ padding: '10px', textAlign: 'center', background: 'rgba(46,204,113,0.12)', borderRadius: '6px', color: '#2ecc71', fontWeight: 700 }}>
                      {cm.TN?.toLocaleString()}<div style={{ fontSize: '0.7rem', color: '#888', fontWeight: 400 }}>correctly approved</div>
                    </td>
                    <td style={{ padding: '10px', textAlign: 'center', background: 'rgba(243,156,18,0.12)', borderRadius: '6px', color: '#f39c12', fontWeight: 700 }}>
                      {cm.FP?.toLocaleString()}<div style={{ fontSize: '0.7rem', color: '#888', fontWeight: 400 }}>good customers sent to review</div>
                    </td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px', color: '#a18cd1', fontWeight: 700 }}>Actually Defaulted</td>
                    <td style={{ padding: '10px', textAlign: 'center', background: 'rgba(231,76,60,0.12)', borderRadius: '6px', color: '#e74c3c', fontWeight: 700 }}>
                      {cm.FN?.toLocaleString()}<div style={{ fontSize: '0.7rem', color: '#888', fontWeight: 400 }}>missed defaults</div>
                    </td>
                    <td style={{ padding: '10px', textAlign: 'center', background: 'rgba(46,204,113,0.12)', borderRadius: '6px', color: '#2ecc71', fontWeight: 700 }}>
                      {cm.TP?.toLocaleString()}<div style={{ fontSize: '0.7rem', color: '#888', fontWeight: 400 }}>defaults caught</div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Operating point + calibration */}
            <div style={{ flex: '1 1 260px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {[
                { k: 'Recall (defaults caught)', v: op.recall != null ? `${(op.recall * 100).toFixed(1)}%` : '—' },
                { k: 'Precision (of those flagged)', v: op.precision != null ? `${(op.precision * 100).toFixed(1)}%` : '—' },
                { k: 'F1 at threshold', v: op.f1 ?? '—' },
                { k: 'Brier before calibration', v: cal.oof_brier_raw ?? '—' },
                { k: 'Brier after calibration', v: cal.oof_brier_calibrated ?? '—' },
                {
                  k: 'Mean predicted vs actual',
                  v: cal.mean_calibrated_probability != null
                    ? `${(cal.mean_calibrated_probability * 100).toFixed(2)}% vs ${(cal.actual_default_rate * 100).toFixed(2)}%`
                    : '—',
                },
              ].map(row => (
                <div key={row.k} style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.04)', borderRadius: '8px', padding: '9px 13px' }}>
                  <span style={{ color: '#aaa', fontSize: '0.82rem' }}>{row.k}</span>
                  <span style={{ color: '#a18cd1', fontWeight: 700, fontSize: '0.85rem' }}>{row.v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Feature Importance Chart */}
      <div className="glass-panel" style={{ marginBottom: '24px' }}>
        <h3>Top 15 Features by LightGBM Split Count</h3>
        <p style={{ color: '#888', fontSize: '0.82rem', marginTop: 0 }}>Higher = more decision splits made on this feature across all trees</p>
        <div style={{ marginTop: '16px', height: '500px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 150, right: 20 }}>
              <XAxis type="number" stroke="#aaa" />
              <YAxis dataKey="name" type="category" stroke="#aaa" width={140} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'rgba(20,20,40,0.95)', border: '1px solid #444', borderRadius: '8px' }}
                cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                formatter={(v: any) => [v, 'Split Count']}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {chartData.map((_, i) => <Cell key={i} fill={GRADIENT_COLORS[i % GRADIENT_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="glass-panel">
        <h3 style={{ marginBottom: '16px' }}>Feature Descriptions</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          {chartData.slice(0, 10).map((item: any, i: number) => (
            <div key={i} style={{
              background: 'rgba(255,255,255,0.04)', borderRadius: '10px', padding: '14px',
              borderLeft: `3px solid ${GRADIENT_COLORS[i]}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ color: '#fff', fontWeight: 600, fontSize: '0.9rem' }}>#{i+1} {item.name}</div>
                <div style={{ color: GRADIENT_COLORS[i], fontWeight: 700, fontSize: '1rem' }}>{item.value}</div>
              </div>
              <div style={{ color: '#888', fontSize: '0.78rem', marginTop: '4px' }}>
                {FEATURE_DESC[item.name] || 'Feature importance by split count'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const FEATURE_DESC: Record<string, string> = {
  'DAYS_ID_PUBLISH':       'Days since ID was last published — proxy for identity stability',
  'DAYS_LAST_PHONE_CHANGE':'Days since phone number changed — frequent changes indicate instability',
  'DAYS_REGISTRATION':     'Days since registration — how long applicant has been in the system',
  'bureau_avg_days_credit':'Average days since bureau credit opened — credit history length',
  'credit_goods_ratio':    'AMT_CREDIT / AMT_GOODS_PRICE — ratio > 1.2 flags cash-out risk',
  'ext_source_mean':       'Average of 3 external credit bureau scores — strongest risk signal',
  'AMT_ANNUITY':           'Monthly loan repayment amount — annuity burden indicator',
  'EXT_SOURCE_3':          'External credit score from source 3 — bureau quality signal',
  'ext_source_min':        'Minimum external credit score across 3 sources — worst-case signal',
  'EXT_SOURCE_2':          'External credit score from source 2 — bureau quality signal',
};

export default Explainability;
