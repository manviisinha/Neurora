import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { FamilyProvider } from "./contexts/FamilyContext";
import LandingPage from "./pages/LandingPage";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import Prescriptions from "./pages/Prescriptions";
import ConflictEngine from "./pages/ConflictEngine";
import LabAnalytics from "./pages/LabAnalytics";
import Pricing from "./pages/Pricing";
import ExpertLocator from "./pages/ExpertLocator";
import Subscription from "./pages/Subscription";
import NotFound from "./pages/NotFound";
import Settings from "./pages/Settings";
const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <FamilyProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/auth" element={<Auth />} />
            <Route element={<AppLayout />}>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/prescriptions" element={<Prescriptions />} />
              <Route path="/conflicts" element={<ConflictEngine />} />
              <Route path="/lab-analytics" element={<LabAnalytics />} />
              <Route path="/pricing" element={<Pricing />} />
              <Route path="/expert-locator" element={<ExpertLocator />} />
              <Route path="/subscription" element={<Subscription />} />
              <Route path="/settings" element={<Settings />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </FamilyProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
