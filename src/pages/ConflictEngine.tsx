import { useState, useEffect } from "react";
import { ShieldAlert, AlertTriangle, CheckCircle, Info, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { db, auth } from "@/lib/firebase";
import { collection, query, onSnapshot } from "firebase/firestore";
import { useFamily } from "@/contexts/FamilyContext";

import { DRUG_DATABASE } from "@/services/medicationKnowledgeBase";

type Severity = "high" | "moderate" | "low" | "none";

const getInteraction = (d1: string, d2: string) => {
  const drug1 = Object.values(DRUG_DATABASE).find(d => d.name.toLowerCase() === d1.toLowerCase());
  if (!drug1) return null;

  const interaction = drug1.interactions.find(i => i.drug.toLowerCase() === d2.toLowerCase());
  if (interaction) return interaction;

  // Check reciprocal
  const drug2 = Object.values(DRUG_DATABASE).find(d => d.name.toLowerCase() === d2.toLowerCase());
  if (!drug2) return null;

  const reciprocal = drug2.interactions.find(i => i.drug.toLowerCase() === d1.toLowerCase());
  return reciprocal || null;
};

const severityStyles: Record<Severity | string, string> = {
  high: "bg-destructive/20 text-destructive border-destructive/30",
  moderate: "bg-neurora-amber/10 text-neurora-amber border-neurora-amber/30",
  low: "bg-neurora-mint/10 text-neurora-mint border-neurora-mint/30",
  none: "bg-secondary/30 text-muted-foreground border-border/20",
};

const cellColor = (s: string | null) => {
  if (!s) return "bg-secondary/20";
  if (s === "high") return "bg-destructive/30";
  if (s === "moderate") return "bg-neurora-amber/20";
  if (s === "low") return "bg-neurora-mint/15";
  return "bg-secondary/20";
};

export default function ConflictEngine() {
  const [userDrugs, setUserDrugs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [focusedConflict, setFocusedConflict] = useState<string | null>(null);
  const { activeMember } = useFamily();

  useEffect(() => {
    if (!auth.currentUser || !activeMember) return;

    const path = activeMember.isPrimary
      ? `users/${auth.currentUser.uid}/medications`
      : `users/${auth.currentUser.uid}/family_members/${activeMember.id}/medications`;

    const q = query(collection(db, path));
    setLoading(true);

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const meds = snapshot.docs.map(doc => doc.data().name);
      // Deduplicate and filter small names
      const uniqueMeds = Array.from(new Set(meds)).filter(m => m && typeof m === 'string' && m.length > 2);
      setUserDrugs(uniqueMeds);
      setLoading(false);
    }, (error) => {
      console.error("Error fetching meds for conflict check:", error);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [activeMember]);

  const hasHighRisk = focusedConflict !== null;
  const drugs = userDrugs.length > 0 ? userDrugs : ["No Active Medications"];

  const activeInteractions: { key: string; d1: string; d2: string; severity: string; desc: string }[] = [];

  if (userDrugs.length > 1) {
    for (let i = 0; i < userDrugs.length; i++) {
      for (let j = i + 1; j < userDrugs.length; j++) {
        const d1 = userDrugs[i];
        const d2 = userDrugs[j];
        const interaction = getInteraction(d1, d2);
        if (interaction) {
          activeInteractions.push({
            key: `${d1}-${d2}`,
            d1,
            d2,
            severity: interaction.severity,
            desc: interaction.description
          });
        }
      }
    }
  }

  return (
    <div className="relative space-y-6 animate-fade-in-up">
      {/* Screen dim overlay for high-risk focus */}
      {hasHighRisk && (
        <div className="fixed inset-0 bg-background/60 z-10 transition-opacity" onClick={() => setFocusedConflict(null)} />
      )}

      <div className="relative z-20">
        <h1 className="text-2xl font-display font-bold text-foreground">Conflict Engine</h1>
        <p className="text-sm text-muted-foreground mt-1">Drug-drug interaction analysis</p>
      </div>

      {/* High Risk Alerts */}
      <div className={cn("space-y-3", hasHighRisk && "relative z-20")}>
        {loading ? (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Loader2 className="w-8 h-8 animate-spin mb-2" />
            <p className="text-sm italic">Analyzing medications for conflicts...</p>
          </div>
        ) : activeInteractions.length > 0 ? (
          activeInteractions.map((val) => {
            const isFocused = focusedConflict === val.key;
            return (
              <div
                key={val.key}
                onClick={() => setFocusedConflict(isFocused ? null : val.key)}
                className={cn(
                  "glass-card p-4 border cursor-pointer transition-all",
                  severityStyles[val.severity],
                  isFocused && "glow-danger scale-[1.01]",
                  !isFocused && val.severity === "high" && "animate-pulse-glow"
                )}
              >
                <div className="flex items-center gap-3">
                  <ShieldAlert className="w-5 h-5 shrink-0" />
                  <div>
                    <p className="font-display font-semibold">{val.d1} × {val.d2}</p>
                    <p className="text-sm opacity-80 mt-1">{val.desc}</p>
                  </div>
                </div>
                {isFocused && (
                  <div className="mt-3 p-3 rounded-lg bg-background/50 text-sm">
                    <p className="flex items-start gap-2"><AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" /> <strong>Action:</strong> Consult with prescribing physician. High interaction risk detected.</p>
                  </div>
                )}
              </div>
            );
          })
        ) : (
          <div className="glass-card p-8 border-dashed border text-center text-muted-foreground">
            <CheckCircle className="w-10 h-10 mx-auto mb-3 opacity-20" />
            <p className="font-medium">No major conflicts detected among your active medications.</p>
          </div>
        )}
      </div>

      {/* Interaction Matrix */}
      <div className={cn("glass-card p-5 overflow-auto", hasHighRisk ? "relative z-0 opacity-40" : "")}>
        <h2 className="text-sm font-display font-semibold text-muted-foreground uppercase tracking-wider mb-4">Interaction Matrix</h2>
        <div className="min-w-[500px]">
          {userDrugs.length > 0 ? (
            <table className="w-full text-xs font-display">
              <thead>
                <tr>
                  <th className="p-2 text-left text-muted-foreground" />
                  {drugs.map((d) => (
                    <th key={d} className="p-2 text-center text-muted-foreground font-medium">{d}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {drugs.map((d1, i) => (
                  <tr key={d1}>
                    <td className="p-2 text-muted-foreground font-medium">{d1}</td>
                    {drugs.map((d2, j) => {
                      if (i === j) return <td key={d2} className="p-2 text-center"><span className="inline-block w-6 h-6 rounded bg-muted/30">—</span></td>;
                      const interaction = getInteraction(d1, d2);
                      return (
                        <td key={d2} className="p-2 text-center">
                          <span className={cn("inline-flex items-center justify-center w-8 h-8 rounded-lg text-xs font-semibold", cellColor(interaction?.severity || null))}>
                            {interaction?.severity === "high" && <AlertTriangle className="w-3.5 h-3.5" />}
                            {interaction?.severity === "moderate" && <Info className="w-3.5 h-3.5" />}
                            {interaction?.severity === "low" && <CheckCircle className="w-3.5 h-3.5" />}
                            {!interaction && "·"}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="py-8 text-center text-muted-foreground italic">
              Add medications to see the interaction matrix.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
