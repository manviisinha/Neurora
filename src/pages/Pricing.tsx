import { Award, Truck, Clock, Check } from "lucide-react";
import { cn } from "@/lib/utils";

const medicines = [
  {
    name: "Metformin 500mg",
    pharmacies: [
      { name: "MedPlus", price: 4.99, delivery: "Same Day", neuroraChoice: true },
      { name: "Apollo Pharmacy", price: 5.49, delivery: "Next Day", neuroraChoice: false },
      { name: "Wellness Forever", price: 6.25, delivery: "2-3 Days", neuroraChoice: false },
    ],
  },
  {
    name: "Lisinopril 10mg",
    pharmacies: [
      { name: "PharmEasy", price: 8.99, delivery: "Next Day", neuroraChoice: false },
      { name: "MedPlus", price: 7.50, delivery: "Same Day", neuroraChoice: true },
      { name: "Netmeds", price: 9.25, delivery: "2-3 Days", neuroraChoice: false },
    ],
  },
  {
    name: "Atorvastatin 20mg",
    pharmacies: [
      { name: "1mg", price: 12.99, delivery: "Next Day", neuroraChoice: false },
      { name: "MedPlus", price: 11.50, delivery: "Same Day", neuroraChoice: true },
      { name: "Apollo Pharmacy", price: 13.75, delivery: "Same Day", neuroraChoice: false },
    ],
  },
];

const priceTier = (price: number) => {
  if (price < 6) return { label: "Budget", color: "bg-neurora-mint/10 text-neurora-mint border-neurora-mint/30" };
  if (price < 10) return { label: "Mid-Range", color: "bg-neurora-amber/10 text-neurora-amber border-neurora-amber/30" };
  return { label: "Premium", color: "bg-primary/10 text-primary border-primary/30" };
};

export default function Pricing() {
  return (
    <div className="space-y-6 animate-fade-in-up">
      <div>
        <h1 className="text-2xl font-display font-bold text-foreground">Medicine Pricing</h1>
        <p className="text-sm text-muted-foreground mt-1">Compare prices across pharmacies</p>
      </div>

      {medicines.map((med) => (
        <div key={med.name} className="glass-card p-5">
          <h2 className="font-display font-semibold text-foreground mb-4">{med.name}</h2>
          <div className="grid gap-3 sm:grid-cols-3">
            {med.pharmacies.map((p, i) => {
              const tier = priceTier(p.price);
              return (
                <div
                  key={i}
                  className={cn(
                    "rounded-xl border p-4 transition-all relative",
                    p.neuroraChoice
                      ? "border-primary/40 bg-primary/5 glow-indigo"
                      : "border-border/20 bg-secondary/20"
                  )}
                >
                  {p.neuroraChoice && (
                    <div className="absolute -top-2.5 left-3 flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary text-primary-foreground text-xs font-display font-semibold">
                      <Award className="w-3 h-3" />
                      Neurora Choice
                    </div>
                  )}
                  <p className="font-display font-semibold text-foreground mt-1">{p.name}</p>
                  <p className="text-2xl font-display font-bold text-foreground mt-2">${p.price.toFixed(2)}</p>

                  <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1"><Truck className="w-3 h-3" />{p.delivery}</span>
                    <span className={cn("px-2 py-0.5 rounded-full border text-xs font-semibold", tier.color)}>{tier.label}</span>
                  </div>

                  {p.neuroraChoice && (
                    <div className="flex items-center gap-1 mt-3 text-xs text-neurora-mint">
                      <Check className="w-3 h-3" />
                      Best value for price &amp; speed
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
