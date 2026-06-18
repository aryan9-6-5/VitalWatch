import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Navigation from "@/components/Navigation";
import Home from "@/pages/home";
import DoctorDashboard from "@/pages/doctor-dashboard";
import FamilyDashboard from "@/pages/family-dashboard";
import Emergency from "@/pages/emergency";
import Pharmacy from "@/pages/pharmacy";
import Pricing from "@/pages/pricing";
import AIAssistant from "@/pages/ai-assistant";
import Vitals from "@/pages/vitals";
import Alerts from "@/pages/alerts";
import PatientAnalysis from "@/pages/patient-analysis";
import NotFound from "@/pages/not-found";

function Router() {
  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      <Switch>
        <Route path="/" component={Home} />
        <Route path="/doctor-dashboard" component={DoctorDashboard} />
        <Route path="/family-dashboard" component={FamilyDashboard} />
        <Route path="/emergency" component={Emergency} />
        <Route path="/pharmacy" component={Pharmacy} />
        <Route path="/pricing" component={Pricing} />
        <Route path="/ai-assistant" component={AIAssistant} />
        <Route path="/vitals" component={Vitals} />
        <Route path="/alerts" component={Alerts} />
        <Route path="/patient-analysis/:id?" component={PatientAnalysis} />
        <Route component={NotFound} />
      </Switch>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
