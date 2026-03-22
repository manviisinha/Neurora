import { useState } from "react";
import { Outlet } from "react-router-dom";
import { NeuralSidebar } from "./NeuralSidebar";
import { TopHeader } from "./TopHeader";
import { SOSButton } from "../SOSButton";
import { AuraChatbot } from "../AuraChatbot";

export function AppLayout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="flex min-h-screen w-full bg-background">
      <NeuralSidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />
      <div className="flex flex-1 flex-col min-w-0">
        <TopHeader onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)} />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
      <SOSButton />
      <AuraChatbot />
    </div>
  );
}
