import { Navigate, Route, Routes } from 'react-router-dom';
import { Boundary } from './components/Boundary';
import { Sidebar } from './components/Sidebar';
import { Overview } from './pages/Overview';
import { Tables } from './pages/Tables';
import { Console } from './pages/Console';
import { TableDetail } from './pages/TableDetail';
import { TableRecords } from './pages/TableRecords';
import { TableSearch } from './pages/TableSearch';
import { TableQuality } from './pages/TableQuality';
import { TableVisualize } from './pages/TableVisualize';
import { RecordDetail } from './pages/RecordDetail';

export function App() {
  return (
    <div className="shell">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <Sidebar />
      <main className="main" id="main">
        {/* Per page, not per app: a crash in one view leaves the navigation usable. */}
        <Boundary>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/tables" element={<Tables />} />
          <Route path="/console" element={<Console />} />
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
        </Boundary>
      </main>
    </div>
  );
}
