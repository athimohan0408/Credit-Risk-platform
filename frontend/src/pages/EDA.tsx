import { useEffect, useState } from 'react';
import api from '../api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell, PieChart, Pie,
} from 'recharts';

const COLORS_INCOME = ['#e74c3c','#e67e22','#f39c12','#3498db','#2ecc71','#9b59b6','#1abc9c','#95a5a6'];
const COLORS_HOUSING = ['#e74c3c','#e67e22','#f39c12','#3498db','#2ecc71','#9b59b6'];

const MetricCard = ({ label, value, sub }: { label: string; value: string; sub?: string }) => (
  <div className="glass-card" style={{ flex: 1, minWidth: '160px' }}>
    <div className="metric-label">{label}</div>
    <div className="metric-value">{value}</div>
    {sub && <div style={{ color: '#888', fontSize: '0.78rem', marginTop: '4px' }}>{sub}</div>}
  </div>
);

const EDA = () => {
  const [data, setData] = useState<any>(null);
  const [activeInsight, setActiveInsight] = useState<number | null>(null);

  useEffect(() => {
    api.get('/eda/summary').then(res => setData(res.data)).catch(console.error);
  }, []);

  if (!data) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh' }}>
      <div style={{ textAlign: 'center' }}>
        <div className="spinner" />
        <p style={{ color: '#aaa', marginTop: '16px' }}>Loading EDA Dashboard...</p>
      </div>
    </div>
  );

  const incomeData = Object.entries(data.income_type_defaults || {}).map(([name, value]) => ({
    name, value,
  })).sort((a: any, b: any) => b.value - a.value);

  const housingData = Object.entries(data.housing_type_defaults || {}).map(([name, value]) => ({
    name, value,
  })).sort((a: any, b: any) => b.value - a.value);

  const featureData = Object.entries(data.metrics?.top_features || {})
    .map(([name, value]) => ({ name, value }))
    .sort((a: any, b: any) => b.value - a.value)
    .slice(0, 10);

  const pieData = [
    { name: 'No Default', value: 91.93, fill: '#2ecc71' },
    { name: 'Default', value: 8.07, fill: '#e74c3c' },
  ];

  const cm = data.metrics?.confusion_matrix || {};

  return (
    <div>
      <div style={{ marginBottom: '8px' }}>
        <h1 style={{ marginBottom: '4px' }}>📊 Exploratory Data Analysis</h1>
        <p style={{ color: '#aaa', marginTop: 0 }}>Home Credit Default Risk — 307,511 applicants</p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: '16px', marginBottom: '28px', flexWrap: 'wrap' }}>
        <MetricCard label="Total Applicants" value={data.total_applicants.toLocaleString()} sub="Training set" />
        <MetricCard label="Default Rate" value={`${data.class_balance.default}%`} sub="8.07% = 24,825 defaults" />
        <MetricCard label="CV ROC-AUC" value={data.metrics?.cv_roc_auc || '0.7651'} sub="5-fold stratified CV" />
        <MetricCard label="CV PR-AUC" value={data.metrics?.cv_pr_auc || '0.2525'} sub="Imbalanced baseline" />
        <MetricCard label="Features" value={`${data.metrics?.n_features || 138}`} sub="After engineering" />
      </div>

      {/* Class Balance + Confusion Matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
        <div className="glass-panel">
          <h3 style={{ marginBottom: '16px' }}>Class Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}%`}>
                {pieData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip formatter={(v: any) => `${v}%`} contentStyle={{ background: 'rgba(20,20,40,0.95)', border: '1px solid #444', borderRadius: '8px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="glass-panel">
          <h3 style={{ marginBottom: '16px' }}>Confusion Matrix (threshold=0.5)</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '12px' }}>
            {[
              { label: 'True Negative', value: cm.TN?.toLocaleString() || '218,589', color: '#2ecc71' },
              { label: 'False Positive', value: cm.FP?.toLocaleString() || '64,097', color: '#e74c3c' },
              { label: 'False Negative', value: cm.FN?.toLocaleString() || '9,642', color: '#e67e22' },
              { label: 'True Positive', value: cm.TP?.toLocaleString() || '15,183', color: '#3498db' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: 'rgba(255,255,255,0.05)', borderRadius: '10px', padding: '14px', border: `1px solid ${color}33`, textAlign: 'center' }}>
                <div style={{ color, fontSize: '1.5rem', fontWeight: 700 }}>{value}</div>
                <div style={{ color: '#aaa', fontSize: '0.78rem', marginTop: '4px' }}>{label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Default Rate by Income Type */}
      <div className="glass-panel" style={{ marginBottom: '24px' }}>
        <h3>Default Rate by Income Type (%)</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={incomeData} margin={{ left: 0, bottom: 30 }}>
            <XAxis dataKey="name" stroke="#aaa" angle={-30} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
            <YAxis stroke="#aaa" tickFormatter={(v) => `${v}%`} />
            <Tooltip
              formatter={(v: any) => [`${v}%`, 'Default Rate']}
              contentStyle={{ background: 'rgba(20,20,40,0.95)', border: '1px solid #444', borderRadius: '8px' }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {incomeData.map((_, i) => <Cell key={i} fill={COLORS_INCOME[i % COLORS_INCOME.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Default Rate by Housing Type */}
      <div className="glass-panel" style={{ marginBottom: '24px' }}>
        <h3>Default Rate by Housing Type (%)</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={housingData} layout="vertical" margin={{ left: 120 }}>
            <XAxis type="number" stroke="#aaa" tickFormatter={(v) => `${v}%`} />
            <YAxis dataKey="name" type="category" stroke="#aaa" width={110} tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(v: any) => [`${v}%`, 'Default Rate']}
              contentStyle={{ background: 'rgba(20,20,40,0.95)', border: '1px solid #444', borderRadius: '8px' }}
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {housingData.map((_, i) => <Cell key={i} fill={COLORS_HOUSING[i % COLORS_HOUSING.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Feature Importances */}
      {featureData.length > 0 && (
        <div className="glass-panel" style={{ marginBottom: '24px' }}>
          <h3>Top 10 Features by LightGBM Importance</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={featureData} layout="vertical" margin={{ left: 160 }}>
              <XAxis type="number" stroke="#aaa" />
              <YAxis dataKey="name" type="category" stroke="#aaa" width={150} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: 'rgba(20,20,40,0.95)', border: '1px solid #444', borderRadius: '8px' }}
                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              />
              <Bar dataKey="value" fill="url(#purpleGrad)" radius={[0, 4, 4, 0]} />
              <defs>
                <linearGradient id="purpleGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#a18cd1" />
                  <stop offset="100%" stopColor="#fbc2eb" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Business Insights */}
      <div className="glass-panel">
        <h3 style={{ marginBottom: '16px' }}>💡 Business Insights</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {(data.insights || []).map((insight: string, i: number) => (
            <div
              key={i}
              onClick={() => setActiveInsight(activeInsight === i ? null : i)}
              style={{
                background: activeInsight === i ? 'rgba(161,140,209,0.12)' : 'rgba(255,255,255,0.04)',
                borderLeft: '3px solid #a18cd1',
                borderRadius: '8px',
                padding: '12px 16px',
                cursor: 'pointer',
                transition: 'background 0.2s',
                fontSize: '0.92rem',
                lineHeight: 1.5,
              }}
            >
              <span style={{ color: '#a18cd1', fontWeight: 700, marginRight: '8px' }}>#{i + 1}</span>
              {insight}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default EDA;
