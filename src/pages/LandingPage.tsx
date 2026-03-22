import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Brain, ShieldCheck, ScanLine, FlaskConical, MapPin, Users, Zap, ArrowRight, CheckCircle2, Star, Crown, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { HeroVisualization } from "@/components/HeroVisualization";
import { auth } from "@/lib/firebase";
import { onAuthStateChanged, User as FirebaseUser } from "firebase/auth";

const features = [
  {
    icon: ScanLine,
    title: "Prescription Parsing",
    desc: "AI-powered OCR extracts medications, dosages, and frequencies from any prescription image in seconds.",
  },
  {
    icon: ShieldCheck,
    title: "Conflict Detection",
    desc: "Real-time drug-drug interaction analysis with severity scoring and safety recommendations.",
  },
  {
    icon: FlaskConical,
    title: "Lab Analytics",
    desc: "Trend analysis and plain-language insights from your lab results, tracked over time.",
  },
  {
    icon: MapPin,
    title: "Expert Locator",
    desc: "Find top-rated, board-certified specialists near you with verified credentials and instant booking.",
    premium: true,
  },
  {
    icon: Users,
    title: "Family Nexus",
    desc: "Manage health profiles for your entire family — spouse, kids, parents — from a single dashboard.",
    premium: true,
  },
  {
    icon: Zap,
    title: "Emergency SOS",
    desc: "Persistent emergency button with AI-powered context detection for critical health situations.",
  },
];

const tiers = [
  {
    name: "Basic",
    price: "Free",
    period: "",
    features: ["3 Prescription Scans / month", "Basic Conflict Detection", "Lab Result Viewing", "Aura AI Chat"],
    cta: "Get Started",
    highlight: false,
  },
  {
    name: "Neurora Plus",
    price: "$19",
    period: "/month",
    features: ["Unlimited Prescription Scans", "Advanced Conflict + Allergy Sync", "Full Lab Analytics & Trends", "Expert Locator Access", "Up to 5 Family Profiles", "Priority Aura AI"],
    cta: "Start Free Trial",
    highlight: true,
  },
];

const testimonials = [
  { name: "Dr. Sarah Chen", role: "Cardiologist", quote: "Neurora's conflict detection has become an essential part of my daily workflow.", rating: 5 },
  { name: "Michael Torres", role: "Patient", quote: "Managing my parents' medications alongside mine has never been easier.", rating: 5 },
  { name: "Dr. Priya Patel", role: "Pharmacist", quote: "The prescription parsing accuracy is remarkable. It saves us hours every week.", rating: 5 },
];

import { useFamily } from "@/contexts/FamilyContext";

export default function LandingPage() {
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const location = useLocation();
  const showProfile = user && location.state?.fromDashboard;
  const { activeMember } = useFamily(); // Use same source of truth as Dashboard

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
      {/* Nav */}
      <header className="fixed top-0 inset-x-0 z-50 border-b border-border/20" style={{ background: "hsl(var(--background) / 0.8)", backdropFilter: "blur(20px)" }}>
        <div className="max-w-7xl mx-auto flex items-center justify-between h-16 px-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/20 flex items-center justify-center glow-indigo">
              <Brain className="w-5 h-5 text-primary" />
            </div>
            <span className="font-display text-xl font-bold text-gradient-indigo tracking-tight">Neurora</span>
          </div>
          <nav className="hidden md:flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#features" className="hover:text-foreground transition-colors">Features</a>
            <a href="#pricing" className="hover:text-foreground transition-colors">Pricing</a>
            <a href="#testimonials" className="hover:text-foreground transition-colors">Testimonials</a>
          </nav>
          <div className="flex items-center gap-3">
            {showProfile ? (
              <Button variant="ghost" size="icon" asChild className="relative overflow-hidden group rounded-full">
                <Link to="/dashboard">
                  <div
                    className="w-9 h-9 rounded-full border-2 border-background flex items-center justify-center relative hover:ring-2 ring-primary/20 transition-all"
                    style={{ backgroundColor: activeMember?.accentColor || "hsla(var(--primary) / 0.2)" }}
                  >
                    <span className="text-sm font-display font-bold text-white">{activeMember?.avatar || "??"}</span>
                    <div className="absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full bg-neurora-gold flex items-center justify-center">
                      <Crown className="w-2.5 h-2.5 text-white" />
                    </div>
                  </div>
                </Link>
              </Button>
            ) : (
              <>
                <Button variant="ghost" size="sm" asChild>
                  <Link to="/auth">Log In</Link>
                </Button>
                <Button size="sm" className="glow-indigo" asChild>
                  <Link to="/auth?mode=signup">Get Started</Link>
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative pt-32 pb-20 px-6">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] rounded-full opacity-20" style={{ background: "radial-gradient(circle, hsl(var(--neurora-indigo) / 0.4), transparent 70%)" }} />
        </div>
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center relative">
          <div className="space-y-8 animate-fade-in-up">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-neurora-gold/30 bg-neurora-gold/5 text-neurora-gold text-xs font-display font-semibold uppercase tracking-wider">
              <Crown className="w-3.5 h-3.5" />
              AI-Powered Medical Intelligence
            </div>
            <h1 className="text-5xl lg:text-6xl font-display font-bold leading-tight">
              Your Health,{" "}
              <span className="text-gradient-indigo">Decoded by AI</span>
            </h1>
            <p className="text-lg text-muted-foreground font-body leading-relaxed max-w-lg">
              Neurora transforms complex medical data into clear, actionable insights. From prescription parsing to drug conflict detection — powered by advanced AI.
            </p>
            <div className="flex flex-wrap gap-4">
              {user ? (
                <Button size="lg" className="glow-indigo text-base px-8" asChild>
                  <Link to="/dashboard">
                    Go to Dashboard <ArrowRight className="w-4 h-4 ml-1" />
                  </Link>
                </Button>
              ) : (
                <Button size="lg" className="glow-indigo text-base px-8" asChild>
                  <Link to="/auth?mode=signup">
                    Get Started <ArrowRight className="w-4 h-4 ml-1" />
                  </Link>
                </Button>
              )}
              <Button variant="outline" size="lg" className="text-base px-8" asChild>
                <a href="#pricing">View Plans</a>
              </Button>
            </div>
            <div className="flex items-center gap-6 pt-2 text-sm text-muted-foreground">
              <span className="flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4 text-neurora-mint" /> Free to start</span>
              <span className="flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4 text-neurora-mint" /> No credit card</span>
              <span className="flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4 text-neurora-mint" /> HIPAA Ready</span>
            </div>
          </div>


          <div className="flex justify-center items-center animate-fade-in-up" style={{ animationDelay: "0.15s" }}>
            <HeroVisualization />
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl lg:text-4xl font-display font-bold">
              Everything You Need for{" "}
              <span className="text-gradient-indigo">Smarter Health</span>
            </h2>
            <p className="mt-4 text-muted-foreground max-w-2xl mx-auto">
              Six powerful AI engines working together to keep you and your family safe.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f) => (
              <div key={f.title} className="glass-card p-6 hover:border-primary/30 transition-colors group">
                <div className="w-12 h-12 rounded-lg bg-secondary/50 flex items-center justify-center mb-4 group-hover:bg-primary/10 transition-colors">
                  <f.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-display font-bold mb-2 flex items-center gap-2">
                  {f.title}
                  {f.premium && <Crown className="w-3.5 h-3.5 text-neurora-gold" />}
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {f.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tiers */}
      <section id="pricing" className="py-20 px-6 bg-secondary/20">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl lg:text-4xl font-display font-bold">
              Simple, Transparent{" "}
              <span className="text-gradient-indigo">Pricing</span>
            </h2>
          </div>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {tiers.map((tier) => (
              <div key={tier.name} className={`glass-card p-8 flex flex-col relative ${tier.highlight ? 'border-primary/40 shadow-lg shadow-primary/5' : ''}`}>
                {tier.highlight && (
                  <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/4 bg-primary text-primary-foreground text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider">
                    Most Popular
                  </div>
                )}
                <h3 className="text-xl font-display font-bold mb-2">{tier.name}</h3>
                <div className="flex items-baseline gap-1 mb-6">
                  <span className="text-4xl font-bold">{tier.price}</span>
                  <span className="text-muted-foreground">{tier.period}</span>
                </div>
                <ul className="space-y-3 mb-8 flex-1">
                  {tier.features.map((feat) => (
                    <li key={feat} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <CheckCircle2 className="w-4 h-4 text-primary shrink-0 mt-0.5" />
                      {feat}
                    </li>
                  ))}
                </ul>
                <Button variant={tier.highlight ? "default" : "outline"} className={tier.highlight ? "glow-indigo" : ""} asChild>
                  <Link to="/auth?mode=signup">{tier.cta}</Link>
                </Button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl lg:text-4xl font-display font-bold">
              Trusted by Professionals
            </h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {testimonials.map((t) => (
              <div key={t.name} className="glass-card p-6">
                <div className="flex gap-1 mb-4">
                  {[...Array(t.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-neurora-gold text-neurora-gold" />
                  ))}
                </div>
                <p className="text-sm text-muted-foreground italic mb-4">"{t.quote}"</p>
                <div>
                  <p className="font-bold text-sm">{t.name}</p>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">{t.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-border/20 bg-background">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center">
              <Brain className="w-4 h-4 text-muted-foreground" />
            </div>
            <span className="font-display font-bold text-muted-foreground">Neurora</span>
          </div>
          <p className="text-sm text-muted-foreground">© 2024 Neurora Health AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
