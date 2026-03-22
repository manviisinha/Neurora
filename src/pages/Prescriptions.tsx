import { useState, useEffect } from "react";
import { Plus, Search, Filter, FileText, Pill, AlertTriangle, Clock, Calendar, Check, MoreVertical, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PrescriptionUpload } from "@/components/PrescriptionUpload";
import { useFamily } from "@/contexts/FamilyContext";
import { collection, query, onSnapshot, deleteDoc, doc, orderBy } from "firebase/firestore";
import { db, auth } from "@/lib/firebase";
import { useToast } from "@/hooks/use-toast";

interface Medication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  description?: string;
  category?: string;
  prescribedAt: unknown;
  status: "active" | "discontinued";
}

export default function Prescriptions() {
  const [showUpload, setShowUpload] = useState(false);
  const [medications, setMedications] = useState<Medication[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  const { activeMember } = useFamily();
  const { toast } = useToast();

  useEffect(() => {
    if (!auth.currentUser || !activeMember) return;

    // Determine path
    let collectionPath = "";
    if (activeMember.isPrimary) {
      collectionPath = `users/${auth.currentUser.uid}/medications`;
    } else {
      collectionPath = `users/${auth.currentUser.uid}/family_members/${activeMember.id}/medications`;
    }

    const q = query(collection(db, collectionPath), orderBy("prescribedAt", "desc"));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const p = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      })) as Medication[];
      setMedications(p);
      setLoading(false);
    }, (error) => {
      console.error("Error fetching meds:", error);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [activeMember]);

  const handleDelete = async (id: string) => {
    if (!auth.currentUser || !activeMember) return;
    try {
      let collectionPath = "";
      if (activeMember.isPrimary) {
        collectionPath = `users/${auth.currentUser.uid}/medications`;
      } else {
        collectionPath = `users/${auth.currentUser.uid}/family_members/${activeMember.id}/medications`;
      }
      await deleteDoc(doc(db, collectionPath, id));
      toast({ title: "Medication Removed", description: "The medication has been deleted." });
    } catch (error: unknown) {
      const err = error as any;
      console.error("Delete error:", error);
      toast({ title: "Error", description: err.message || "Could not delete medication.", variant: "destructive" });
    }
  };

  const filteredMeds = medications.filter(m =>
    m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    m.category?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-foreground">Prescriptions</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {activeMember?.isPrimary ? "Manage your medications" : `Medications for ${activeMember?.name}`}
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => setShowUpload(!showUpload)}
            className={showUpload ? "bg-secondary text-foreground hover:bg-secondary/80" : "glow-indigo"}
          >
            {showUpload ? "Cancel Upload" : <><Plus className="w-4 h-4 mr-2" /> Add Prescription</>}
          </Button>
        </div>
      </div>

      {showUpload && (
        <div className="mb-8">
          <PrescriptionUpload onSuccess={() => setShowUpload(false)} />
        </div>
      )}

      {/* Search & Stats */}
      {!showUpload && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="md:col-span-2 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search medications..."
              className="pl-9 bg-background/50 border-border/40"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="glass-card p-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-neurora-mint/10 text-neurora-mint"><Pill className="w-4 h-4" /></div>
              <span className="text-sm font-medium">Active</span>
            </div>
            <span className="text-lg font-bold">{medications.length}</span>
          </div>
          <div className="glass-card p-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-neurora-amber/10 text-neurora-amber"><AlertTriangle className="w-4 h-4" /></div>
              <span className="text-sm font-medium">Conflicts</span>
            </div>
            <span className="text-lg font-bold">0</span>
          </div>
        </div>
      )}

      {/* Medication List */}
      <div className="grid gap-4">
        {loading ? (
          <div className="text-center py-12 text-muted-foreground">Loading prescriptions...</div>
        ) : filteredMeds.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground italic">
            No medications found. Upload a prescription to get started.
          </div>
        ) : (
          filteredMeds.map((med) => (
            <div key={med.id} className="glass-card p-5 flex flex-col md:flex-row md:items-center gap-4 group hover:border-primary/30 transition-all">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-display font-bold text-lg text-foreground">{med.name}</h3>
                  {(med as any).tag && (
                    <span className="text-[9px] px-2 py-0.5 rounded-md bg-primary/10 text-primary uppercase font-bold tracking-tight">
                      {(med as any).tag}
                    </span>
                  )}
                  {med.category && <span className="text-[10px] px-2 py-0.5 rounded-full bg-secondary text-muted-foreground">{med.category}</span>}
                </div>
                <p className="text-sm text-muted-foreground line-clamp-1 italic">{med.description || "Active prescription"}</p>

                <div className="flex items-center gap-3 mt-2 text-[10px] text-muted-foreground uppercase tracking-widest font-medium">
                  <Calendar className="w-3 h-3" />
                  {(med as any).dateStr || (med.prescribedAt ? (med.prescribedAt as any).toDate().toLocaleDateString() : "New Entry")}
                </div>
              </div>

              <div className="flex items-center gap-6 text-sm text-white/80">
                <div className="flex items-center gap-1.5 min-w-[100px] bg-secondary/30 px-3 py-1.5 rounded-lg border border-white/5">
                  <Pill className="w-3.5 h-3.5 text-primary/70" />
                  <span className="font-medium">{med.dosage}</span>
                </div>
                <div className="flex items-center gap-1.5 min-w-[120px] bg-secondary/30 px-3 py-1.5 rounded-lg border border-white/5">
                  <Clock className="w-3.5 h-3.5 text-neurora-indigo/70" />
                  <span className="font-medium">{med.frequency}</span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-destructive/50 hover:text-destructive hover:bg-destructive/10 transition-colors"
                  onClick={() => handleDelete(med.id)}
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
