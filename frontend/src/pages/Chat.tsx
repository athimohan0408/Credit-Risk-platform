import React, { useState, useRef, useEffect } from 'react';
import api from '../api';

const EXAMPLE_QUERIES = [
  "What is the average annual income of applicants?",
  "What is the default rate by gender?",
  "Show top 5 applicants with highest credit who defaulted",
  "How many applicants own a car and defaulted?",
  "What is the average EXT_SOURCE_2 for defaulters vs non-defaulters?",
  "Count applicants by income type",
];

const Chat = () => {
  const [query, setQuery]   = useState('');
  const [messages, setMessages] = useState<any[]>([]);
  const [loading, setLoading]   = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async (q: string) => {
    if (!q.trim()) return;
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', text: q }]);
    setLoading(true);
    try {
      const res = await api.post('/chat', { query: q });
      setMessages(prev => [...prev, { role: 'bot', data: res.data }]);
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'bot', error: err?.response?.data?.detail || err.message || 'An error occurred.' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); sendMessage(query); };

  const fmt = (v: any) => {
    if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4);
    return String(v);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 80px)' }}>
      <h1 style={{ marginBottom: '4px' }}>💬 Talk to Data</h1>
      <p style={{ color: '#aaa', marginTop: 0, marginBottom: '12px' }}>
        Natural-language → SQL chatbot powered by Google Gemini · SELECT-only · SQLite backend
      </p>

      {/* Example chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
        {EXAMPLE_QUERIES.map((q, i) => (
          <button
            key={i}
            onClick={() => sendMessage(q)}
            disabled={loading}
            style={{
              background: 'rgba(161,140,209,0.12)', border: '1px solid rgba(161,140,209,0.3)',
              color: '#c9b8f0', borderRadius: '20px', padding: '5px 14px',
              fontSize: '0.78rem', cursor: 'pointer', transition: 'all 0.15s',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = 'rgba(161,140,209,0.25)'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = 'rgba(161,140,209,0.12)'; }}
          >
            {q}
          </button>
        ))}
      </div>

      {/* Chat window */}
      <div className="glass-panel chat-container" style={{ flex: 1, overflowY: 'auto', marginBottom: '14px', padding: '16px' }}>
        {messages.length === 0 && (
          <div style={{ color: '#555', textAlign: 'center', marginTop: '60px', fontSize: '0.9rem' }}>
            <div style={{ fontSize: '2rem', marginBottom: '8px' }}>🗃️</div>
            Ask any question about the 307,511 Home Credit applicants.<br />
            Click an example above or type your own question.
          </div>
        )}

        {messages.map((msg, idx) => (
          msg.role === 'user' ? (
            <div key={idx} style={{ display: 'flex', justifyContent: 'flex-end', margin: '8px 0' }}>
              <div className="user-bubble">{msg.text}</div>
            </div>
          ) : (
            <div key={idx} style={{ display: 'flex', justifyContent: 'flex-start', margin: '8px 0' }}>
              <div className="bot-bubble" style={{ maxWidth: '90%', width: '90%' }}>
                {msg.error ? (
                  <span style={{ color: '#e74c3c' }}>❌ {msg.error}</span>
                ) : (
                  <>
                    {/* Status badge */}
                    <div style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                      <span style={{
                        padding: '2px 10px', borderRadius: '12px', fontSize: '0.75rem', fontWeight: 700,
                        background: msg.data.status === 'success' ? '#2ecc7122' : '#e74c3c22',
                        color: msg.data.status === 'success' ? '#2ecc71' : '#e74c3c',
                        border: `1px solid ${msg.data.status === 'success' ? '#2ecc71' : '#e74c3c'}44`,
                      }}>
                        {msg.data.status?.toUpperCase()}
                      </span>
                      <span style={{ color: '#888', fontSize: '0.8rem' }}>{msg.data.message}</span>
                    </div>

                    {/* SQL */}
                    {msg.data.sql && (
                      <div style={{
                        background: 'rgba(0,0,0,0.4)', padding: '10px 14px', borderRadius: '8px',
                        fontFamily: 'monospace', fontSize: '0.82rem', color: '#fbc2eb',
                        marginBottom: '10px', overflowX: 'auto', border: '1px solid rgba(255,255,255,0.08)'
                      }}>
                        {msg.data.sql}
                      </div>
                    )}

                    {/* Results table */}
                    {msg.data.result?.length > 0 && (
                      <div style={{ overflowX: 'auto' }}>
                        <div style={{ color: '#888', fontSize: '0.75rem', marginBottom: '6px' }}>
                          📊 {msg.data.row_count} row{msg.data.row_count !== 1 ? 's' : ''} returned
                        </div>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                          <thead>
                            <tr>
                              {Object.keys(msg.data.result[0]).map((k: string) => (
                                <th key={k} style={{
                                  textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.15)',
                                  padding: '6px 10px', color: '#a18cd1', fontWeight: 700, whiteSpace: 'nowrap'
                                }}>{k}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {msg.data.result.slice(0, 20).map((row: any, i: number) => (
                              <tr key={i} style={{ background: i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent' }}>
                                {Object.values(row).map((v: any, j: number) => (
                                  <td key={j} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', padding: '5px 10px', color: '#ddd' }}>
                                    {fmt(v)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {msg.data.row_count > 20 && (
                          <div style={{ color: '#666', fontSize: '0.75rem', marginTop: '6px' }}>
                            Showing 20 of {msg.data.row_count} rows
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )
        ))}

        {loading && (
          <div style={{ display: 'flex', justifyContent: 'flex-start', margin: '8px 0' }}>
            <div className="bot-bubble">
              <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
                <span style={{ color: '#888', fontSize: '0.85rem' }}>Thinking</span>
                {[0,1,2].map(i => (
                  <div key={i} style={{
                    width: '7px', height: '7px', borderRadius: '50%', background: '#a18cd1',
                    animation: `bounce 1.2s ${i * 0.2}s infinite ease-in-out`,
                  }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          className="glass-input"
          placeholder="Ask a question about the data..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          disabled={loading}
          style={{ flex: 1, fontSize: '0.95rem' }}
        />
        <button type="submit" className="glass-button" disabled={loading || !query.trim()} style={{ whiteSpace: 'nowrap' }}>
          Send ➤
        </button>
      </form>
    </div>
  );
};

export default Chat;
