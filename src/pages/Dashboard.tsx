import { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Minus, User, Phone, Mail, Calendar, AlertTriangle, Droplets, Clock, ExternalLink, ShieldCheck, Activity, Atom } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { MoleculeVisualization } from "@/components/MoleculeVisualization";
import { auth, db } from "@/lib/firebase";
import { doc, getDoc, onSnapshot, query, collection } from "firebase/firestore";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Info, Pill } from "lucide-react";
import { onAuthStateChanged } from "firebase/auth";
import { useFamily } from "@/contexts/FamilyContext";
import { Button } from "@/components/ui/button";
import { DRUG_DATABASE } from "@/services/medicationKnowledgeBase";

// Hardcoded data for visualization
const healthData = [
  { month: "Jan", value: 72, zone: "stable" },
  { month: "Feb", value: 75, zone: "improved" },
  { month: "Mar", value: 71, zone: "stable" },
  { month: "Apr", value: 68, zone: "degraded" },
  { month: "May", value: 74, zone: "improved" },
  { month: "Jun", value: 78, zone: "improved" },
  { month: "Jul", value: 76, zone: "stable" }];

interface Conflict {
  drug1: string;
  drug2: string;
  severity: "high" | "moderate" | "low";
  desc: string;
}

const severityColor = (s: string) => {
  if (s === "high") return "bg-destructive/20 border-destructive/40 text-destructive";
  if (s === "moderate") return "bg-neurora-amber/10 border-neurora-amber/30 text-neurora-amber";
  return "bg-neurora-mint/10 border-neurora-mint/30 text-neurora-mint";
};

interface PatientData {
  full_name?: string;
  email?: string;
  phone?: string;
  blood_group?: string;
  date_of_birth?: string;
  medical_conditions?: string;
  isManaged?: boolean;
}

interface Medication {
  id?: string;
  name: string;
  category?: string;
  interactions?: { drug: string; description: string }[];
  description?: string;
  dosage?: string;
  frequency?: string;
  [key: string]: any; // Allow for other Firestore properties for now, but name is required
}

export default function Dashboard() {
  const [userData, setUserData] = useState<PatientData | null>(null);
  const [loading, setLoading] = useState(true);
  const [medications, setMedications] = useState<Medication[]>([]);
  const [detectedConflicts, setDetectedConflicts] = useState<Conflict[]>([]);
  const [selectedMed, setSelectedMed] = useState<Medication | null>(null);
  const [showAllConflicts, setShowAllConflicts] = useState(false);
  const { activeMember } = useFamily();

  useEffect(() => {
    // If we have no active member yet, wait
    if (!activeMember) return;

    // If pending, no data to fetch yet
    if (activeMember.status === 'pending') {
      setUserData(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    const fetchUserData = async () => {
      try {
        if (activeMember.isPrimary && auth.currentUser) {
          const docRef = doc(db, "users", auth.currentUser.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists()) setUserData(docSnap.data());
        } else if (auth.currentUser) {
          // Fetch sub-profile data (e.g. managed profile)
          const docRef = doc(db, "users", auth.currentUser.uid, "family_members", activeMember.id);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists()) {
            // Merge with basic info
            setUserData({ ...docSnap.data(), isManaged: true });
          } else {
            setUserData({ full_name: activeMember.name, isManaged: true });
          }
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, [activeMember]);

  // Fetch medications and check for conflicts
  useEffect(() => {
    if (!auth.currentUser || !activeMember) return;

    let collectionPath = "";
    if (activeMember.isPrimary) {
      collectionPath = `users/${auth.currentUser.uid}/medications`;
    } else {
      collectionPath = `users/${auth.currentUser.uid}/family_members/${activeMember.id}/medications`;
    }

    const q = query(collection(db, collectionPath));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const meds = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() })) as Medication[];
      setMedications(meds);

      // Dynamic Conflict Detection
      const conflictsList: Conflict[] = [];
      const medNames = meds.map((m) => m.name.toLowerCase());

      meds.forEach((med) => {
        const drugInfo = Object.values(DRUG_DATABASE).find(d =>
          d.name.toLowerCase() === med.name.toLowerCase()
        );

        if (drugInfo) {
          drugInfo.interactions.forEach(interaction => {
            if (medNames.includes(interaction.drug.toLowerCase())) {
              // Avoid duplicate pairs
              const exists = conflictsList.find(c =>
                (c.drug1 === drugInfo.name && c.drug2 === interaction.drug) ||
                (c.drug1 === interaction.drug && c.drug2 === drugInfo.name)
              );

              if (!exists) {
                conflictsList.push({
                  drug1: drugInfo.name,
                  drug2: interaction.drug,
                  severity: interaction.severity,
                  desc: interaction.description
                });
              }
            }
          });
        }
      });

      setDetectedConflicts(conflictsList);
    });

    return () => unsubscribe();
  }, [activeMember]);

  // Render "Pending" State
  if (activeMember?.status === 'pending') {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-6 text-center animate-fade-in-up">
        <div className="w-20 h-20 rounded-full bg-neurora-amber/10 flex items-center justify-center relative">
          <Mail className="w-10 h-10 text-neurora-amber" />
          <span className="absolute top-0 right-0 flex h-4 w-4">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neurora-amber opacity-75"></span>
            <span className="relative inline-flex rounded-full h-4 w-4 bg-neurora-amber"></span>
          </span>
        </div>
        <div className="space-y-2 max-w-md">
          <h2 className="text-2xl font-display font-bold">Invitation Sent</h2>
          <p className="text-muted-foreground">
            An invitation has been sent to <span className="font-semibold text-foreground">{activeMember.email}</span>.
            Once {activeMember.name} accepts, their health data will appear here.
          </p>
        </div>
        <Button variant="outline" className="gap-2">
          <Clock className="w-4 h-4" /> Resend Invite
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold text-foreground">
            {activeMember?.isPrimary ? "Analysis Canvas" : `${activeMember?.name}'s Health`}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {activeMember?.isPrimary ? "Your AI-powered health overview" : `Managed Profile • ${activeMember?.relation}`}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Profile Card - Read Only */}
        <div className="glass-card p-6 lg:col-span-1">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-display font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <User className="w-4 h-4" /> Patient Profile
            </h2>
          </div>

          {loading ? (
            <div className="h-40 flex items-center justify-center text-sm text-muted-foreground">Loading...</div>
          ) : userData ? (
            <div className="space-y-6">
              <div>
                <p className="text-2xl font-bold font-display">{userData.full_name || "Guest Patient"}</p>
                <div className="flex items-center gap-2 text-sm text-muted-foreground mt-1">
                  <Mail className="w-3.5 h-3.5" /> {userData.email || "No email"}
                </div>
                {userData.phone && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mt-1">
                    <Phone className="w-3.5 h-3.5" /> {userData.phone}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-secondary/20 p-2.5 rounded-lg border border-border/20">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 flex items-center gap-1"><Droplets className="w-3 h-3" /> Blood Group</p>
                  <p className="font-semibold">{userData.blood_group || "--"}</p>
                </div>
                <div className="bg-secondary/20 p-2.5 rounded-lg border border-border/20">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 flex items-center gap-1"><Calendar className="w-3 h-3" /> Age</p>
                  <p className="font-semibold text-sm">
                    {userData.date_of_birth ? (() => {
                      const birthDate = new Date(userData.date_of_birth);
                      const today = new Date();
                      let age = today.getFullYear() - birthDate.getFullYear();
                      const m = today.getMonth() - birthDate.getMonth();
                      if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) age--;
                      return isNaN(age) ? "--" : age + " Years";
                    })() : "--"}
                  </p>
                </div>
              </div>

              {userData.medical_conditions && (
                <div className="pt-2 border-t border-border/20">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> Health Status</p>
                  <p className="text-sm text-foreground/90 bg-secondary/10 p-3 rounded-lg border border-border/10 leading-relaxed italic">
                    "{userData.medical_conditions}"
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No data available for this profile yet.
            </div>
          )}
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-2 space-y-6">
          {activeMember?.isPrimary ? (
            <>
              {/* Health Confidence Chart */}
              <div className="glass-card p-6 h-[400px] flex flex-col">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                      <Activity className="w-5 h-5 text-primary" /> Health Timeline
                    </h2>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="flex items-center gap-1 text-neurora-mint"><TrendingUp className="w-4 h-4" /> Stable</span>
                  </div>
                </div>
                <div className="flex-1 w-full min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={healthData}>
                      <defs>
                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                      <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} axisLine={false} domain={[60, 100]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))', borderRadius: '8px' }}
                        itemStyle={{ color: 'hsl(var(--foreground))' }}
                      />
                      <Area
                        type="monotone"
                        dataKey="value"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorValue)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Conflict Shield */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-neurora-gold" /> Conflict Shield
                  </h2>
                  <Button variant="ghost" size="sm" className="text-xs" onClick={() => setShowAllConflicts(true)}>View All</Button>
                </div>
                <div className="space-y-3">
                  {detectedConflicts.length > 0 ? (
                    detectedConflicts.map((conflict, i) => (
                      <div key={i} className="bg-secondary/20 p-3 rounded-lg border border-border/20 flex items-start gap-3 hover:bg-secondary/30 transition-colors">
                        <div className={`mt-1 h-2 w-2 rounded-full shrink-0 ${conflict.severity === 'high' ? 'bg-destructive' :
                          conflict.severity === 'moderate' ? 'bg-neurora-amber' : 'bg-neurora-mint'
                          }`} />
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
                            <p className="text-sm font-semibold">{conflict.drug1} + {conflict.drug2}</p>
                            <span className={`text-[10px] px-2 py-0.5 rounded-full border uppercase tracking-wider font-semibold ${severityColor(conflict.severity)}`}>
                              {conflict.severity}
                            </span>
                          </div>
                          <p className="text-xs text-muted-foreground">{conflict.desc}</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="py-8 text-center border border-dashed border-border/50 rounded-lg">
                      <ShieldCheck className="w-8 h-8 text-neurora-mint mx-auto mb-2 opacity-50" />
                      <p className="text-sm text-muted-foreground">No critical conflicts detected.</p>
                      <p className="text-[10px] text-muted-foreground mt-1">Based on current prescription profile</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Medicine Knowledge Engine */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-display font-semibold flex items-center gap-2">
                    <Atom className="w-5 h-5 text-neurora-indigo" /> Medicine Knowledge Engine
                  </h2>
                </div>
                <div className="flex flex-wrap gap-2 mb-4">
                  {medications.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => {
                        const info = Object.values(DRUG_DATABASE).find(d => d.name.toLowerCase() === m.name.toLowerCase());
                        setSelectedMed(info || { name: m.name, description: "No detailed information available in our local engine yet.", category: m.category });
                      }}
                      className="text-xs px-3 py-1.5 rounded-full bg-secondary/30 border border-border/20 hover:bg-primary/10 hover:border-primary/30 transition-all flex items-center gap-1.5"
                    >
                      <Info className="w-3 h-3" /> {m.name}
                    </button>
                  ))}
                  {medications.length === 0 && <p className="text-xs text-muted-foreground italic">Save a prescription to see clinical insights.</p>}
                </div>
                <div className="h-[250px] w-full bg-secondary/5 rounded-lg overflow-hidden relative border border-border/20 flex items-center justify-center group">
                  <MoleculeVisualization />
                  <div className="absolute inset-0 bg-background/20 backdrop-blur-[2px] opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest">Molecular Structural Analysis Active</p>
                  </div>
                </div>
              </div>

              {/* Detail Dialogs */}
              <Dialog open={!!selectedMed} onOpenChange={() => setSelectedMed(null)}>
                <DialogContent className="glass-card border-border/20">
                  <DialogHeader>
                    <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 text-primary">
                      <Pill className="w-6 h-6" />
                    </div>
                    <DialogTitle className="text-2xl font-display">{selectedMed?.name}</DialogTitle>
                    <DialogDescription className="text-neurora-indigo font-medium text-xs tracking-wider uppercase">
                      {selectedMed?.category || "Medication"}
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 pt-4">
                    <div className="bg-secondary/20 p-4 rounded-xl border border-border/10">
                      <p className="text-sm leading-relaxed text-foreground/90">{selectedMed?.description}</p>
                    </div>
                    {selectedMed?.interactions && selectedMed.interactions.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Known Interactions</p>
                        {selectedMed.interactions.map((int: { drug: string; description: string }, i: number) => (
                          <div key={i} className="flex items-center gap-2 text-xs bg-destructive/5 p-2 rounded-lg border border-destructive/10">
                            <AlertTriangle className="w-3 h-3 text-destructive" />
                            <span className="font-medium text-destructive">{int.drug}:</span>
                            <span className="text-muted-foreground">{int.description}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </DialogContent>
              </Dialog>

              <Dialog open={showAllConflicts} onOpenChange={setShowAllConflicts}>
                <DialogContent className="max-w-2xl glass-card border-border/20">
                  <DialogHeader>
                    <DialogTitle className="text-xl font-display flex items-center gap-2">
                      <ShieldCheck className="w-5 h-5 text-neurora-gold" /> Detailed Conflict Analysis
                    </DialogTitle>
                    <DialogDescription>
                      Clinical cross-referencing for your current {medications.length} active medications.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-3 mt-4 max-h-[60vh] overflow-y-auto pr-2">
                    {detectedConflicts.map((conflict, i) => (
                      <div key={i} className="bg-secondary/20 p-4 rounded-xl border border-border/20">
                        <div className="flex items-center justify-between mb-2">
                          <p className="font-bold text-foreground flex items-center gap-2">
                            {conflict.drug1} <Minus className="w-3 h-3 opacity-30" /> {conflict.drug2}
                          </p>
                          <span className={`text-[10px] px-2.5 py-1 rounded-full border uppercase tracking-widest font-bold ${severityColor(conflict.severity)}`}>
                            {conflict.severity}
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground leading-relaxed">{conflict.desc}</p>
                        <div className="mt-4 flex gap-2">
                          <Button variant="outline" size="sm" className="h-7 text-[10px] uppercase font-bold tracking-wider">Clinical Notes</Button>
                          <Button variant="ghost" size="sm" className="h-7 text-[10px] uppercase font-bold tracking-wider">Alt. Options</Button>
                        </div>
                      </div>
                    ))}
                    {detectedConflicts.length === 0 && (
                      <div className="text-center py-12 opacity-50 space-y-2">
                        <ShieldCheck className="w-12 h-12 mx-auto text-neurora-mint" />
                        <p className="text-sm">No clinical conflicts detected across your medication profile.</p>
                      </div>
                    )}
                  </div>
                </DialogContent>
              </Dialog>
            </>
          ) : (
            <div className="glass-card p-6 h-full flex flex-col items-center justify-center text-center space-y-4 min-h-[400px]">
              <div className="p-4 rounded-full bg-secondary/30">
                <AlertTriangle className="w-8 h-8 text-muted-foreground" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Health Timeline</h3>
                <p className="text-sm text-muted-foreground max-w-xs mx-auto">
                  Health data for {activeMember?.name} will appear here once populated.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}