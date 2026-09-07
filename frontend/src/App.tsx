import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { BarChart2, ShieldAlert, FileSearch, CheckSquare, MessageSquare } from 'lucide-react';

import EDA from './pages/EDA';
import Predict from './pages/Predict';
import Rules from './pages/Rules';
import Chat from './pages/Chat';
import Explainability from './pages/Explainability';

const Sidebar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'EDA Dashboard', icon: BarChart2 },
    { path: '/predict', label: 'Risk Prediction', icon: ShieldAlert },
    { path: '/explainability', label: 'Explainability', icon: FileSearch },
    { path: '/rules', label: 'Business Rules', icon: CheckSquare },
    { path: '/chat', label: 'Talk to Data', icon: MessageSquare },
  ];

  return (
    <div className="sidebar">
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '30px' }}>
        <h2 style={{ margin: 0, fontSize: '1.2rem' }}>🏦 Credit Risk Platform</h2>
      </div>
      <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '20px' }}>
        Home Credit Default Risk
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            style={{
              padding: '12px 16px',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              background: location.pathname === item.path ? 'rgba(161,140,209,0.2)' : 'transparent',
              color: location.pathname === item.path ? '#fff' : '#aaa',
              borderLeft: location.pathname === item.path ? '3px solid #a18cd1' : '3px solid transparent',
            }}
          >
            <item.icon size={18} />
            {item.label}
          </Link>
        ))}
      </nav>
    </div>
  );
};

function App() {
  return (
    <Router>
      <Sidebar />
      <div className="main-content">
        <Routes>
          <Route path="/" element={<EDA />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/rules" element={<Rules />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
