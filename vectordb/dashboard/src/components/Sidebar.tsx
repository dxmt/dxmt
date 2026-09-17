import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Icon, type IconName } from './Icon';

/**
 * The sidebar, which collapses to icons.
 *
 * Collapsed it is 72px and every label is gone, so each link keeps a `title` — the icon alone
 * is a guess otherwise, and a console is not a place to make people guess. The links stay
 * links: collapsing changes what is drawn, never what is reachable, so Tab still walks the same
 * five destinations in the same order.
 *
 * The choice is remembered, and the remembering is allowed to fail. `localStorage` throws in a
 * private window and returns nothing after site data is cleared, and a sidebar is not worth a
 * blank page — so it is read in a try/catch and the default stands when it does not answer.
 */
const KEY = 'dxmt.sidebar.collapsed';

const LINKS: { to: string; label: string; icon: IconName; end?: boolean }[] = [
  { to: '/', label: 'Overview', icon: 'overview', end: true },
  { to: '/tables', label: 'Tables', icon: 'tables' },
];

function remembered(): boolean {
  try {
    return window.localStorage.getItem(KEY) === '1';
  } catch {
    return false;
  }
}

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(remembered);

  useEffect(() => {
    try {
      window.localStorage.setItem(KEY, collapsed ? '1' : '0');
    } catch {
      // A preference that cannot be saved is still a preference for this session.
    }
  }, [collapsed]);

  return (
    <nav className="nav" data-collapsed={collapsed || undefined} aria-label="Sections">
      <div className="nav-head">
        <span className="brand" title="DXMT">
          DXMT
        </span>
        <button
          type="button"
          className="nav-toggle ghost"
          aria-expanded={!collapsed}
          aria-controls="nav-links"
          title={collapsed ? 'Expand the sidebar' : 'Collapse the sidebar'}
          onClick={() => setCollapsed((on) => !on)}
        >
          <Icon name="collapse" size={17} />
        </button>
      </div>

      <div id="nav-links">
        {LINKS.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            title={link.label}
            className={({ isActive }) => (isActive ? 'on' : '')}
          >
            <Icon name={link.icon} />
            <span className="label">{link.label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
