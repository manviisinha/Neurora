import { useState } from "react";
import { Menu, Search, Bell, Crown, LogOut, User, Settings, Calendar, Pill } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FamilySwitcher } from "@/components/FamilySwitcher";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { auth } from "@/lib/firebase";
import { signOut } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useFamily } from "@/contexts/FamilyContext";

interface Props {
  onToggleSidebar: () => void;
}

const notifications = [
  { id: 1, title: "Refill Reminder", desc: "Metformin 500mg is running low.", time: "2h ago", icon: Pill, color: "text-neurora-amber" },
  { id: 2, title: "Appointment", desc: "Dr. Sarah Chen tomorrow at 10:00 AM.", time: "5h ago", icon: Calendar, color: "text-primary" },
  { id: 3, title: "Lab Results", desc: "Your latest blood work is ready.", time: "1d ago", icon: User, color: "text-neurora-mint" },
];

export function TopHeader({ onToggleSidebar }: Props) {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { activeMember } = useFamily();

  const handleLogout = async () => {
    try {
      await signOut(auth);
      toast({ title: "Logged out", description: "See you soon!" });
      navigate("/");
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  return (
    <header className="h-16 border-b border-border/30 flex items-center justify-between px-6 shrink-0 bg-background/50 backdrop-blur-sm sticky top-0 z-40">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={onToggleSidebar} className="text-muted-foreground hover:text-foreground">
          <Menu className="w-5 h-5" />
        </Button>
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search patients, medicines..."
            className="h-9 w-64 rounded-lg bg-secondary/50 border border-border/30 pl-9 pr-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 transition-colors"
          />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <FamilySwitcher />

        {/* Notifications Popover */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="relative text-muted-foreground hover:text-foreground">
              <Bell className="w-5 h-5" />
              <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-neurora-amber animate-pulse" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-80 p-0 mr-4" align="end">
            <div className="p-3 border-b border-border/30">
              <h4 className="font-display font-semibold text-sm">Notifications</h4>
            </div>
            <div className="max-h-[300px] overflow-y-auto">
              {notifications.map((n) => (
                <div key={n.id} className="p-3 hover:bg-secondary/30 transition-colors flex gap-3 border-b border-border/10 last:border-0 cursor-pointer">
                  <div className={`mt-1 bg-secondary/50 p-1.5 rounded-md h-fit ${n.color}`}>
                    <n.icon className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="text-sm font-medium leading-none mb-1">{n.title}</p>
                    <p className="text-xs text-muted-foreground leading-snug">{n.desc}</p>
                    <p className="text-[10px] text-muted-foreground mt-1.5">{n.time}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="p-2 border-t border-border/30 text-center">
              <Button variant="ghost" size="sm" className="w-full text-xs h-8">View All</Button>
            </div>
          </PopoverContent>
        </Popover>

        {/* User Profile Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="outline-none flex items-center gap-2 hover:bg-secondary/50 p-1.5 rounded-full transition-colors">
              <div
                className="w-9 h-9 rounded-full border-2 border-background flex items-center justify-center relative hover:ring-2 ring-primary/20 transition-all"
                style={{ backgroundColor: activeMember?.accentColor || "hsla(var(--primary) / 0.2)" }}
              >
                <span className="text-sm font-display font-bold text-white">{activeMember?.avatar || "??"}</span>
                <div className="absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full bg-neurora-gold flex items-center justify-center">
                  <Crown className="w-2.5 h-2.5 text-white" />
                </div>
              </div>
              <div className="hidden md:block text-left mr-1">
                <p className="text-xs font-bold font-display leading-tight">{activeMember?.name}</p>
                <p className="text-[10px] text-muted-foreground">Premium User</p>
              </div>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>My Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="cursor-pointer" onClick={() => navigate("/settings")}>
              <Settings className="mr-2 h-4 w-4" />
              <span>Settings</span>
            </DropdownMenuItem>
            <DropdownMenuItem className="cursor-pointer">
              <Crown className="mr-2 h-4 w-4 text-neurora-gold" />
              <span>Subscription</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout} className="text-destructive focus:text-destructive cursor-pointer">
              <LogOut className="mr-2 h-4 w-4" />
              <span>Log out</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
