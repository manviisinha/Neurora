import { NavLink, useLocation, Link } from "react-router-dom";
import {
  LayoutDashboard,
  ScanLine,
  ShieldAlert,
  FlaskConical,
  DollarSign,
  ChevronLeft,
  ChevronRight,
  Brain,
  MapPin,
  Crown,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
  { title: "Prescriptions", path: "/prescriptions", icon: ScanLine },
  { title: "Conflict Engine", path: "/conflicts", icon: ShieldAlert },
  { title: "Lab Analytics", path: "/lab-analytics", icon: FlaskConical },
  { title: "Pricing", path: "/pricing", icon: DollarSign },
];

const premiumItems = [
  { title: "Expert Locator", path: "/expert-locator", icon: MapPin },
  { title: "Subscription", path: "/subscription", icon: Crown },
];

interface Props {
  collapsed: boolean;
  onToggle: () => void;
}

export function NeuralSidebar({ collapsed, onToggle }: Props) {
  const location = useLocation();

  return (
    <aside
      className={cn(
        "glass-sidebar flex flex-col transition-all duration-300 ease-in-out shrink-0",
        collapsed ? "w-[68px]" : "w-[240px]"
      )}
    >
      {/* Logo */}
      <Link
        to="/"
        state={{ fromDashboard: true }}
        className="flex items-center gap-3 px-4 h-16 border-b border-border/30 hover:bg-white/5 transition-colors"
      >
        <div className="w-9 h-9 rounded-lg bg-primary/20 flex items-center justify-center glow-indigo">
          <Brain className="w-5 h-5 text-primary" />
        </div>
        {!collapsed && (
          <span className="font-display text-lg font-bold text-gradient-indigo tracking-tight">
            Neurora
          </span>
        )}
      </Link>

      {/* Nav */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group relative",
                isActive
                  ? "bg-primary/15 text-primary-foreground"
                  : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
              )}
            >
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-6 rounded-r-full bg-primary glow-indigo" />
              )}
              <item.icon
                className={cn(
                  "w-5 h-5 shrink-0 transition-colors",
                  isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                )}
              />
              {!collapsed && (
                <span className={cn("font-medium text-sm", isActive && "text-primary")}>
                  {item.title}
                </span>
              )}
            </NavLink>
          );
        })}

        {/* Premium section */}
        {!collapsed && (
          <p className="px-3 pt-4 pb-1 text-[10px] uppercase tracking-wider text-neurora-gold font-display font-semibold flex items-center gap-1">
            <Crown className="w-3 h-3" /> Premium
          </p>
        )}
        {collapsed && <div className="my-2 mx-3 h-px bg-neurora-gold/20" />}
        {premiumItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group relative",
                isActive
                  ? "bg-neurora-gold/10 text-neurora-gold"
                  : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
              )}
            >
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-6 rounded-r-full bg-neurora-gold glow-gold" />
              )}
              <item.icon
                className={cn(
                  "w-5 h-5 shrink-0 transition-colors",
                  isActive ? "text-neurora-gold" : "text-muted-foreground group-hover:text-foreground"
                )}
              />
              {!collapsed && (
                <span className={cn("font-medium text-sm", isActive && "text-neurora-gold")}>
                  {item.title}
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="mx-2 mb-4 flex items-center justify-center h-9 rounded-lg border border-border/30 text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  );
}
