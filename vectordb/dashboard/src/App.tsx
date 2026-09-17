import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { Overview } from './pages/Overview';
import { Tables } from './pages/Tables';
import { TableDetail } from './pages/TableDetail';
import { TableRecords } from './pages/TableRecords';
import { TableSearch } from './pages/TableSearch';
import { TableQuality } from './pages/TableQuality';
import { TableVisualize } from './pages/TableVisualize';
import { RecordDetail } from './pages/RecordDetail';

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
          <Route path="/tables/:name" element={<TableDetail />}>
            <Route index element={<Navigate to="records" replace />} />
            <Route path="records" element={<TableRecords />} />
            <Route path="search" element={<TableSearch />} />
            <Route path="visualize" element={<TableVisualize />} />
            <Route path="quality" element={<TableQuality />} />
          </Route>
          <Route path="/tables/:name/records/:id" element={<RecordDetail />} />
          <Route path="/tables/:name/*" element={<Navigate to="records" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
