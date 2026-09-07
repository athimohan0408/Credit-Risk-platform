import React, { useState } from 'react';
import api from '../api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';

const RISK_CONFIG = {
  LOW:    { color: '#2ecc71', bg: 'rgba(46,204,113,0.12)', label: '✅ LOW RISK',    pct: 33 },
  MEDIUM: { color: '#f39c12', bg: 'rgba(243,156,18,0.12)',  label: '⚠️ MEDIUM RISK', pct: 66 },
  HIGH:   { color: '#e74c3c', bg: 'rgba(231,76,60,0.12)',   label: '🚨 HIGH RISK',   pct: 100 },
};

const FieldGroup = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div style={{ marginBottom: '18px' }}>
    <div style={{ color: '#888', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '6px' }}>{label}</div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>{children}</div>
  </div>
);

const FormField = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div>
    <label style={{ display: 'block', color: '#bbb', fontSize: '0.82rem', marginBottom: '5px' }}>{label}</label>
    {children}
  </div>
);

const Predict = () => {
  const [form, setForm] = useState({
    AMT_INCOME_TOTAL: 150000,
    AMT_CREDIT: 500000,
    AMT_ANNUITY: 25000,
    AMT_GOODS_PRICE: 450000,
    DAYS_BIRTH: -14600,
    DAYS_EMPLOYED: -2000,
    EXT_SOURCE_1: 0.5,
    EXT_SOURCE_2: 0.5,
    EXT_SOURCE_3: 0.5,
    CODE_GENDER: 'M',
    FLAG_OWN_CAR: 'N',
    FLAG_OWN_REALTY: 'Y',
    NAME_INCOME_TYPE: 'Working',
    NAME_EDUCATION_TYPE: 'Secondary / secondary special',
    NAME_HOUSING_TYPE: 'House / apartment',
    CNT_CHILDREN: 0,
  });
  const [result, setResult] = useState<any>(null);
  const [shapData, setShapData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [explaining, setExplaining] = useState(false);

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm(f => ({ ...f, [k]: isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value) }));

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setShapData(null);
    try {
      const res = await api.post('/predict', form);
      setResult(res.data);
      // Fetch SHAP explanation
      setExplaining(true);
      const shapRes = await api.post('/explain', form);
      setShapData(shapRes.data);
    } catch (err) {
      console.error(err);
      alert('Prediction failed. Make sure the backend is running and the model is trained.');
    } finally {
      setLoading(false);
      setExplaining(false);
    }
  };

  const riskCfg = result ? RISK_CONFIG[result.risk_label as keyof typeof RISK_CONFIG] : null;

  const shapChartData = shapData?.shap_values
    ?.sort((a: any, b: any) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 12)
    .map((d: any) => ({ ...d, name: d.label || d.feature, abs: Math.abs(d.shap_value) })) || [];

  return (
    <div>
      <h1 style={{ marginBottom: '4px' }}>🎯 Risk Prediction</h1>
      <p style={{ color: '#aaa', marginTop: 0 }}>Enter applicant data to predict default probability</p>

      <div style={{ display: 'grid', gridTemplateColumns: '420px 1fr', gap: '24px', marginTop: '20px' }}>

        {/* Left: Form */}
        <div className="glass-panel" style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 160px)' }}>
          <h3 style={{ marginBottom: '16px' }}>Applicant Data</h3>
          <form onSubmit={handlePredict}>
            <FieldGroup label="Financial">
              <FormField label="Annual Income (₽)">
                <input type="number" className="glass-input" value={form.AMT_INCOME_TOTAL} onChange={set('AMT_INCOME_TOTAL')} />
              </FormField>
              <FormField label="Credit Amount (₽)">
                <input type="number" className="glass-input" value={form.AMT_CREDIT} onChange={set('AMT_CREDIT')} />
              </FormField>
              <FormField label="Annuity (₽/month)">
                <input type="number" className="glass-input" value={form.AMT_ANNUITY} onChange={set('AMT_ANNUITY')} />
              </FormField>
              <FormField label="Goods Price (₽)">
                <input type="number" className="glass-input" value={form.AMT_GOODS_PRICE} onChange={set('AMT_GOODS_PRICE')} />
              </FormField>
            </FieldGroup>

            <FieldGroup label="External Credit Scores (0–1)">
              <FormField label="EXT_SOURCE_1">
                <input type="number" step="0.01" min="0" max="1" className="glass-input" value={form.EXT_SOURCE_1} onChange={set('EXT_SOURCE_1')} />
              </FormField>
              <FormField label="EXT_SOURCE_2">
                <input type="number" step="0.01" min="0" max="1" className="glass-input" value={form.EXT_SOURCE_2} onChange={set('EXT_SOURCE_2')} />
              </FormField>
              <FormField label="EXT_SOURCE_3">
                <input type="number" step="0.01" min="0" max="1" className="glass-input" value={form.EXT_SOURCE_3} onChange={set('EXT_SOURCE_3')} />
              </FormField>
            </FieldGroup>

            <FieldGroup label="Personal">
              <FormField label="Age (days negative, e.g. -14600)">
                <input type="number" className="glass-input" value={form.DAYS_BIRTH} onChange={set('DAYS_BIRTH')} />
              </FormField>
              <FormField label="Employment Duration (days neg)">
                <input type="number" className="glass-input" value={form.DAYS_EMPLOYED} onChange={set('DAYS_EMPLOYED')} />
              </FormField>
              <FormField label="Gender">
                <select className="glass-input" value={form.CODE_GENDER} onChange={set('CODE_GENDER')}>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </FormField>
              <FormField label="Children">
                <input type="number" min="0" className="glass-input" value={form.CNT_CHILDREN} onChange={set('CNT_CHILDREN')} />
              </FormField>
              <FormField label="Owns Car">
                <select className="glass-input" value={form.FLAG_OWN_CAR} onChange={set('FLAG_OWN_CAR')}>
                  <option value="N">No</option>
                  <option value="Y">Yes</option>
                </select>
              </FormField>
              <FormField label="Owns Realty">
                <select className="glass-input" value={form.FLAG_OWN_REALTY} onChange={set('FLAG_OWN_REALTY')}>
                  <option value="Y">Yes</option>
                  <option value="N">No</option>
                </select>
              </FormField>
            </FieldGroup>

            <FieldGroup label="Socioeconomic">
              <FormField label="Income Type">
                <select className="glass-input" value={form.NAME_INCOME_TYPE} onChange={set('NAME_INCOME_TYPE')}>
                  {['Working','Pensioner','State servant','Commercial associate','Unemployed','Businessman','Maternity leave'].map(t => <option key={t}>{t}</option>)}
                </select>
              </FormField>
              <FormField label="Education">
                <select className="glass-input" value={form.NAME_EDUCATION_TYPE} onChange={set('NAME_EDUCATION_TYPE')}>
                  {['Secondary / secondary special','Higher education','Incomplete higher','Lower secondary','Academic degree'].map(t => <option key={t}>{t}</option>)}
                </select>
              </FormField>
              <FormField label="Housing Type">
                <select className="glass-input" value={form.NAME_HOUSING_TYPE} onChange={set('NAME_HOUSING_TYPE')}>
                  {['House / apartment','Rented apartment','With parents','Municipal apartment','Office apartment','Co-op apartment'].map(t => <option key={t}>{t}</option>)}
                </select>
              </FormField>
            </FieldGroup>

            <button type="submit" className="glass-button" disabled={loading} style={{ width: '100%', padding: '13px', fontSize: '1rem' }}>
              {loading ? '🔄 Predicting...' : '🎯 Run Prediction'}
            </button>
          </form>
        </div>

        {/* Right: Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* Risk Result */}
          <div className="glass-panel" style={{ textAlign: 'center', minHeight: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            {result && riskCfg ? (
              <>
                <div style={{ fontSize: '0.85rem', color: '#aaa', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                  Probability of Default
                </div>
                <div style={{ fontSize: '3.5rem', fontWeight: 800, color: riskCfg.color, lineHeight: 1 }}>
                  {(result.default_probability * 100).toFixed(1)}%
                </div>

                {result.risk_multiple != null && (
                  <div style={{ marginTop: '6px', fontSize: '0.9rem', color: '#bbb' }}>
                    <strong style={{ color: riskCfg.color }}>{result.risk_multiple}x</strong> the portfolio
                    baseline of {(result.baseline_rate * 100).toFixed(2)}%
                  </div>
                )}

                <div style={{ marginTop: '12px', display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'center' }}>
                  <div style={{ padding: '8px 24px', borderRadius: '24px', background: riskCfg.bg, border: `1.5px solid ${riskCfg.color}`, color: riskCfg.color, fontWeight: 700, fontSize: '1.1rem' }}>
                    {riskCfg.label}
                  </div>
                  {result.decision && (
                    <div style={{
                      padding: '8px 24px', borderRadius: '24px', fontWeight: 700, fontSize: '1.1rem',
                      background: result.decision === 'APPROVE' ? 'rgba(46,204,113,0.12)' : 'rgba(243,156,18,0.12)',
                      border: `1.5px solid ${result.decision === 'APPROVE' ? '#2ecc71' : '#f39c12'}`,
                      color: result.decision === 'APPROVE' ? '#2ecc71' : '#f39c12',
                    }}>
                      {result.decision === 'APPROVE' ? '✅ AUTO-APPROVE' : '🔍 SEND TO REVIEW'}
                    </div>
                  )}
                </div>

                {/* Risk gauge. Scaled to the range the calibrated model actually
                    produces — on a 0–100% axis every applicant sits at the far
                    left and the marker conveys nothing. */}
                <div style={{ marginTop: '22px', width: '100%', maxWidth: '380px' }}>
                  {(() => {
                    const AXIS_MAX = 40; // calibrated probabilities rarely exceed this
                    const pct = result.default_probability * 100;
                    const pos = (v: number) => Math.min(Math.max((v / AXIS_MAX) * 100, 0), 100);
                    return (
                      <>
                        <div style={{ height: '10px', borderRadius: '5px', background: 'linear-gradient(to right, #2ecc71, #f39c12, #e74c3c)', position: 'relative' }}>
                          {/* decision threshold marker */}
                          {result.threshold != null && (
                            <div style={{
                              position: 'absolute', top: '-6px', left: `calc(${pos(result.threshold * 100)}% - 1px)`,
                              width: '2px', height: '22px', background: '#fff', opacity: 0.75,
                            }} title={`Review threshold ${(result.threshold * 100).toFixed(0)}%`} />
                          )}
                          <div style={{
                            position: 'absolute', top: '-4px',
                            left: `calc(${pos(pct)}% - 9px)`,
                            width: '18px', height: '18px', borderRadius: '50%',
                            background: riskCfg.color, border: '3px solid #fff', boxShadow: `0 0 8px ${riskCfg.color}`,
                            transition: 'left 0.5s ease',
                          }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.72rem', color: '#666' }}>
                          <span>0%</span>
                          <span>{(AXIS_MAX / 2).toFixed(0)}%</span>
                          <span>{AXIS_MAX}%+</span>
                        </div>
                        <div style={{ fontSize: '0.72rem', color: '#777', marginTop: '4px' }}>
                          White line = review threshold ({((result.threshold ?? 0) * 100).toFixed(0)}%)
                        </div>
                      </>
                    );
                  })()}
                </div>

                <div style={{ marginTop: '10px', fontSize: '0.75rem', color: '#777' }}>
                  Risk score {result.risk_score}/100 · raw model output {result.raw_model_score} before calibration
                </div>
              </>
            ) : (
              <p style={{ color: '#666' }}>Submit the form to see the prediction result.</p>
            )}
          </div>

          {/* Plain-English reasons */}
          {shapData?.reasons && (
            <div className="glass-panel">
              <h3 style={{ marginBottom: '4px' }}>💬 Why this decision?</h3>
              <p style={{ color: '#888', fontSize: '0.82rem', marginTop: 0, marginBottom: '14px' }}>
                The factors that moved this applicant's score, in plain language.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <div style={{ color: '#e74c3c', fontWeight: 700, fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px' }}>
                    Pushed risk up
                  </div>
                  {shapData.reasons.increases?.map((r: any) => (
                    <div key={r.feature} style={{
                      background: 'rgba(231,76,60,0.08)', borderLeft: '3px solid #e74c3c',
                      borderRadius: '6px', padding: '8px 11px', marginBottom: '7px', fontSize: '0.82rem', color: '#ddd',
                    }}>
                      <span style={{ color: '#e74c3c', fontWeight: 700 }}>{r.share_pct}%</span> — {r.sentence}
                    </div>
                  ))}
                </div>
                <div>
                  <div style={{ color: '#2ecc71', fontWeight: 700, fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px' }}>
                    Pushed risk down
                  </div>
                  {shapData.reasons.decreases?.map((r: any) => (
                    <div key={r.feature} style={{
                      background: 'rgba(46,204,113,0.08)', borderLeft: '3px solid #2ecc71',
                      borderRadius: '6px', padding: '8px 11px', marginBottom: '7px', fontSize: '0.82rem', color: '#ddd',
                    }}>
                      <span style={{ color: '#2ecc71', fontWeight: 700 }}>{r.share_pct}%</span> — {r.sentence}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* SHAP Waterfall */}
          <div className="glass-panel" style={{ flex: 1 }}>
            <h3 style={{ marginBottom: '4px' }}>🔬 SHAP Feature Attribution</h3>
            <p style={{ color: '#888', fontSize: '0.82rem', marginTop: '0', marginBottom: '16px' }}>
              {explaining ? '⏳ Computing SHAP values...' : shapData ? 'Top features driving this prediction' : 'Will appear after prediction'}
            </p>
            {shapChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={shapChartData} layout="vertical" margin={{ left: 140, right: 20 }}>
                  <XAxis type="number" stroke="#aaa" tickFormatter={(v) => v.toFixed(3)} />
                  <YAxis dataKey="name" type="category" stroke="#aaa" width={130} tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(_v: any, _n: any, props: any) => [props.payload.shap_value.toFixed(4), 'SHAP Value']}
                    contentStyle={{ background: 'rgba(20,20,40,0.95)', border: '1px solid #444', borderRadius: '8px' }}
                    cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                  />
                  <ReferenceLine x={0} stroke="#666" strokeDasharray="3 3" />
                  <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                    {shapChartData.map((d: any, i: number) => (
                      <Cell key={i} fill={d.shap_value > 0 ? '#e74c3c' : '#2ecc71'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#555', textAlign: 'center', paddingTop: '60px', fontSize: '0.9rem' }}>
                SHAP waterfall chart will appear here after prediction.
              </div>
            )}
            {shapData && (
              <div style={{ marginTop: '12px', fontSize: '0.78rem', color: '#888' }}>
                🔴 Red bars = increases default risk &nbsp;|&nbsp; 🟢 Green bars = decreases default risk
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predict;
