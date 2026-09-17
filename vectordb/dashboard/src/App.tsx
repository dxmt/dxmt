import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { Overview } from './pages/Overview';
import { Tables } from './pages/Tables';

export function App() {
  return (
    <div className="shell">
      <nav className="nav">
        <h1>DXMT VectorDB</h1>
        <NavLink to="/" end className={({ isActive }) => (isActive ? 'on' : '')}>
          Overview
        </NavLink>
        <NavLink to="/tables" className={({ isActive }) => (isActive ? 'on' : '')}>
          Tables
        </NavLink>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/tables" element={<Tables />} />
          {/* Table detail, search, visualize and quality land here next. Until then a deep
              link should go somewhere rather than render nothing. */}
          <Route path="/tables/:name/*" element={<Navigate to="/tables" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
