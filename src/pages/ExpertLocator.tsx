import { useState } from "react";
import { MapPin, Star, Clock, Shield, Calendar, Lock, Search } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

const doctors = [
  { id: 1, name: "Dr. Ananya Sharma", specialty: "Cardiologist", rating: 4.9, reviews: 342, distance: 1.2, availability: "Today", verified: true, sentiment: 96, image: "AS" },
  { id: 2, name: "Dr. Vikram Mehta", specialty: "Endocrinologist", rating: 4.8, reviews: 218, distance: 2.5, availability: "Tomorrow", verified: true, sentiment: 93, image: "VM" },
  { id: 3, name: "Dr. Kavita Rao", specialty: "Neurologist", rating: 4.7, reviews: 189, distance: 3.8, availability: "Today", verified: true, sentiment: 91, image: "KR" },
  { id: 4, name: "Dr. Sanjay Gupta", specialty: "General Physician", rating: 4.9, reviews: 512, distance: 0.8, availability: "Today", verified: true, sentiment: 97, image: "SG" },
  { id: 5, name: "Dr. Priya Nair", specialty: "Dermatologist", rating: 4.6, reviews: 156, distance: 5.1, availability: "Wed", verified: true, sentiment: 89, image: "PN" },
  { id: 6, name: "Dr. Arjun Reddy", specialty: "Orthopedist", rating: 4.8, reviews: 267, distance: 4.2, availability: "Thu", verified: true, sentiment: 94, image: "AR" },
];

const specialties = ["All", "Cardiologist", "Neurologist", "Endocrinologist", "General Physician", "Dermatologist", "Orthopedist"];

export default function ExpertLocator() {
  const [maxDistance, setMaxDistance] = useState([10]);
  const [selectedSpecialty, setSelectedSpecialty] = useState("All");
  const [isPremium] = useState(true);

  const filtered = doctors
    .filter((d) => d.distance <= maxDistance[0])
    .filter((d) => selectedSpecialty === "All" || d.specialty === selectedSpecialty)
    .sort((a, b) => a.distance - b.distance);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold text-foreground flex items-center gap-2">
            Elite Care Network
            <span className="text-xs px-2 py-0.5 rounded-full bg-neurora-gold/15 text-neurora-gold border border-neurora-gold/30 font-semibold">PREMIUM</span>
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Board-certified specialists near you</p>
        </div>
      </div>

      {/* Filters */}
      <div className="glass-card-premium p-5">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="text-xs font-display font-semibold text-muted-foreground uppercase tracking-wider mb-3 block">
              Distance: {maxDistance[0]} km
            </label>
            <Slider value={maxDistance} onValueChange={setMaxDistance} max={15} min={1} step={0.5} className="w-full" />
          </div>
          <div>
            <label className="text-xs font-display font-semibold text-muted-foreground uppercase tracking-wider mb-3 block">Specialty</label>
            <div className="flex flex-wrap gap-2">
              {specialties.map((s) => (
                <button
                  key={s}
                  onClick={() => setSelectedSpecialty(s)}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-display font-semibold transition-all",
                    selectedSpecialty === s
                      ? "bg-neurora-amethyst/20 text-neurora-amethyst-light border border-neurora-amethyst/40"
                      : "bg-secondary/30 text-muted-foreground border border-border/20 hover:bg-secondary/50"
                  )}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Live Map Integration */}
      <div className="glass-card-premium p-1 relative overflow-hidden rounded-xl h-[400px] border border-neurora-gold/20 shadow-lg shadow-neurora-gold/5">
        <iframe
          width="100%"
          height="100%"
          frameBorder="0"
          scrolling="no"
          marginHeight={0}
          marginWidth={0}
          src={`https://maps.google.com/maps?q=${encodeURIComponent(selectedSpecialty + " doctors near me")}&t=&z=${maxDistance[0] <= 3 ? 15 : maxDistance[0] <= 7 ? 14 : maxDistance[0] <= 12 ? 13 : 12}&ie=UTF8&iwloc=&output=embed`}
          className="rounded-lg grayscale-[20%] hover:grayscale-0 transition-all duration-700 contrast-125 opacity-90 hover:opacity-100"
          title="Live Expert Map"
        ></iframe>
        <div className="absolute bottom-4 left-4 right-4 pointer-events-none">
          <div className="bg-background/80 backdrop-blur-md p-3 rounded-lg border border-border/20 shadow-sm flex items-center gap-3">
            <div className="relative">
              <span className="absolute inline-flex h-3 w-3 rounded-full bg-neurora-mint opacity-75 animate-ping"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-neurora-mint"></span>
            </div>
            <div>
              <p className="text-xs font-bold text-foreground">Live Updates Active</p>
              <p className="text-[10px] text-muted-foreground">Showing real-time availability for {selectedSpecialty}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Doctor Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filtered.map((doc) => (
          <div
            key={doc.id}
            className={cn(
              "glass-card-premium p-5 group hover:border-neurora-gold/40 transition-all duration-300",
              !isPremium && "blur-sm pointer-events-none select-none"
            )}
          >
            <div className="flex items-start gap-4">
              <div className="w-14 h-14 rounded-xl bg-neurora-amethyst/20 border border-neurora-amethyst/30 flex items-center justify-center text-lg font-display font-bold text-neurora-amethyst-light shrink-0">
                {doc.image}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="font-display font-semibold text-foreground text-sm">{doc.name}</h3>
                  {doc.verified && <Shield className="w-3.5 h-3.5 text-neurora-gold" />}
                </div>
                <p className="text-xs text-neurora-amethyst-light">{doc.specialty}</p>
                <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Star className="w-3 h-3 text-neurora-gold" /> {doc.rating}
                    <span className="text-muted-foreground/60">({doc.reviews})</span>
                  </span>
                  <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> {doc.distance} km</span>
                </div>
              </div>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center gap-3 text-xs">
                <span className="flex items-center gap-1 text-neurora-mint">
                  <Clock className="w-3 h-3" /> {doc.availability}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-neurora-mint/10 border border-neurora-mint/20 text-neurora-mint text-[10px] font-semibold">
                  {doc.sentiment}% Positive
                </span>
              </div>
              <button className="px-3 py-1.5 rounded-lg bg-neurora-gold/15 border border-neurora-gold/30 text-neurora-gold text-xs font-display font-semibold hover:bg-neurora-gold/25 transition-colors flex items-center gap-1">
                <Calendar className="w-3 h-3" /> Book
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Blurred CTA for non-premium */}
      {!isPremium && (
        <div className="absolute inset-0 flex items-center justify-center z-20">
          <div className="glass-card-premium p-8 text-center max-w-md">
            <Lock className="w-10 h-10 text-neurora-gold mx-auto mb-4" />
            <h3 className="font-display font-bold text-lg text-foreground mb-2">Unlock Expert Access</h3>
            <p className="text-sm text-muted-foreground mb-4">Register to view 15+ top-rated specialists in your city</p>
            <button className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-neurora-gold to-neurora-amethyst text-white font-display font-semibold text-sm glow-gold">
              Upgrade to Neurora Plus
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
