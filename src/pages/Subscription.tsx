import { Check, X, Crown, Sparkles, Lock, Zap, Users, MapPin, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";

const features = [
  { name: "AI Prescription OCR", free: "Limited (3/mo)", premium: "Unlimited", icon: Zap },
  { name: "Conflict Detection", free: "Basic", premium: "Advanced + Allergy Sync", icon: ShieldCheck },
  { name: "Family Profiles", free: false, premium: "Up to 5 Members", icon: Users },
  { name: "Doctor Locator", free: false, premium: "Geo-Verified Specialists", icon: MapPin },
  { name: "Lab Report Analysis", free: "Basic Trends", premium: "Full AI Insights", icon: Sparkles },
  { name: "Priority Support", free: false, premium: "24/7 Chat", icon: Crown },
];

export default function Subscription() {
  return (
    <div className="space-y-8 animate-fade-in-up max-w-4xl mx-auto">
      <div className="text-center">
        <h1 className="text-3xl font-display font-bold text-foreground">
          Choose Your <span className="text-gradient-gold">Plan</span>
        </h1>
        <p className="text-sm text-muted-foreground mt-2">Unlock the full power of AI-driven healthcare</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Free Tier */}
        <div className="glass-card p-6 flex flex-col">
          <div className="mb-6">
            <p className="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground">Basic</p>
            <div className="flex items-baseline gap-1 mt-2">
              <span className="text-4xl font-display font-bold text-foreground">Free</span>
            </div>
            <p className="text-sm text-muted-foreground mt-2">Essential AI analysis to get started</p>
          </div>

          <div className="space-y-3 flex-1">
            {features.map((f) => (
              <div key={f.name} className="flex items-center gap-3">
                {f.free ? (
                  <Check className="w-4 h-4 text-neurora-mint shrink-0" />
                ) : (
                  <X className="w-4 h-4 text-muted-foreground/40 shrink-0" />
                )}
                <div>
                  <p className={cn("text-sm", f.free ? "text-foreground" : "text-muted-foreground/50")}>{f.name}</p>
                  {typeof f.free === "string" && <p className="text-xs text-muted-foreground">{f.free}</p>}
                </div>
              </div>
            ))}
          </div>

          <button className="mt-6 w-full py-3 rounded-xl border border-border/30 bg-secondary/30 text-foreground font-display font-semibold text-sm hover:bg-secondary/50 transition-colors">
            Current Plan
          </button>
        </div>

        {/* Premium Tier */}
        <div className="glass-card-premium p-6 flex flex-col relative overflow-hidden">
          <div className="absolute top-0 right-0 px-4 py-1.5 rounded-bl-xl bg-gradient-to-r from-neurora-gold to-neurora-amethyst">
            <span className="text-xs font-display font-bold text-white flex items-center gap-1">
              <Crown className="w-3 h-3" /> RECOMMENDED
            </span>
          </div>

          <div className="mb-6">
            <p className="text-xs font-display font-semibold uppercase tracking-wider text-neurora-gold">Neurora Plus</p>
            <div className="flex items-baseline gap-1 mt-2">
              <span className="text-4xl font-display font-bold text-gradient-gold">$9.99</span>
              <span className="text-sm text-muted-foreground">/month</span>
            </div>
            <p className="text-sm text-muted-foreground mt-2">Complete family health management</p>
          </div>

          <div className="space-y-3 flex-1">
            {features.map((f) => (
              <div key={f.name} className="flex items-center gap-3">
                <div className="w-5 h-5 rounded-full bg-neurora-gold/15 flex items-center justify-center shrink-0">
                  <Check className="w-3 h-3 text-neurora-gold" />
                </div>
                <div>
                  <p className="text-sm text-foreground">{f.name}</p>
                  <p className="text-xs text-neurora-amethyst-light">{f.premium}</p>
                </div>
              </div>
            ))}
          </div>

          <button className="mt-6 w-full py-3 rounded-xl bg-gradient-to-r from-neurora-gold to-neurora-amethyst text-white font-display font-bold text-sm glow-gold hover:opacity-90 transition-opacity flex items-center justify-center gap-2">
            <Lock className="w-4 h-4" /> Unlock Neurora Plus
          </button>
        </div>
      </div>

      {/* Feature comparison table */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-display font-semibold text-muted-foreground uppercase tracking-wider mb-4">Detailed Comparison</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/30">
                <th className="text-left py-3 px-4 font-display font-semibold text-muted-foreground">Feature</th>
                <th className="text-center py-3 px-4 font-display font-semibold text-muted-foreground">Basic</th>
                <th className="text-center py-3 px-4 font-display font-semibold text-neurora-gold">Neurora Plus</th>
              </tr>
            </thead>
            <tbody>
              {features.map((f) => (
                <tr key={f.name} className="border-b border-border/10">
                  <td className="py-3 px-4 flex items-center gap-2 text-foreground">
                    <f.icon className="w-4 h-4 text-muted-foreground" /> {f.name}
                  </td>
                  <td className="py-3 px-4 text-center text-muted-foreground">
                    {f.free === false ? <X className="w-4 h-4 mx-auto text-muted-foreground/30" /> : f.free}
                  </td>
                  <td className="py-3 px-4 text-center text-neurora-amethyst-light">{f.premium}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
