import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/ui/runs", label: "Runs" },
  { to: "/ui/judge", label: "Judge" },
  { to: "/ui/gate", label: "Gate" },
  { to: "/ui/registry", label: "Registry" },
];

export function Layout() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="title">PROMPTLINE</div>
          <div className="version">v0.1.0</div>
        </div>
        <nav>
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
