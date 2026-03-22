import { useState, useRef, useEffect } from "react";
import { useFamily } from "@/contexts/FamilyContext";
import { ChevronDown, Crown, Check, Plus, Trash2, UserPlus, X, Mail } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

export function FamilySwitcher() {
  const { members, activeMember, setActiveMember, addMember, removeMember, simulateAcceptInvite, loading } = useFamily();
  const [open, setOpen] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);

  const [newMemberName, setNewMemberName] = useState("");
  const [newMemberRelation, setNewMemberRelation] = useState("");
  const [newMemberEmail, setNewMemberEmail] = useState("");

  const ref = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setShowAddForm(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const handleAddMember = async () => {
    if (!newMemberName || !newMemberRelation) return;
    try {
      await addMember(newMemberName, newMemberRelation, newMemberEmail);
      setNewMemberName("");
      setNewMemberRelation("");
      setNewMemberEmail("");
      setShowAddForm(false);
      toast({
        title: newMemberEmail ? "Invite Sent" : "Member Added",
        description: newMemberEmail
          ? `Invitation sent to ${newMemberEmail}`
          : `${newMemberName} added to family.`
      });
    } catch (error: unknown) {
      const err = error as any;
      console.error("Add member error:", error);
      toast({
        title: "Error adding member",
        description: err.message || "Please check your connection and try again.",
        variant: "destructive"
      });
    }
  };

  const handleRemoveMember = async (e: React.MouseEvent, id: string, name: string) => {
    e.stopPropagation();
    try {
      await removeMember(id);
      toast({ title: "Member Removed", description: `${name} removed from family.` });
    } catch (error) {
      toast({ title: "Error", description: "Could not remove member.", variant: "destructive" });
    }
  };

  const handleSimulateAccept = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await simulateAcceptInvite(id);
      toast({ title: "Invite Accepted", description: "This user is now active (Simulated)." });
    } catch (error) {
      toast({ title: "Error", description: "Could not accept invite.", variant: "destructive" });
    }
  };

  if (loading || !activeMember) return <div className="w-24 h-8 bg-secondary/30 rounded-lg animate-pulse" />;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-secondary/50 transition-colors"
      >
        {/* Avatar cluster */}
        <div className="flex -space-x-2">
          {members.slice(0, 3).map((m) => (
            <div
              key={m.id}
              className={cn(
                "w-7 h-7 rounded-full border-2 border-background flex items-center justify-center text-[10px] font-display font-bold",
                m.id === activeMember.id ? "ring-2 ring-neurora-gold z-10" : "opacity-60"
              )}
              style={{ backgroundColor: m.accentColor }}
            >
              <span className="text-white">{m.avatar}</span>
            </div>
          ))}
          {members.length > 3 && (
            <div className="w-7 h-7 rounded-full border-2 border-background bg-secondary flex items-center justify-center text-[10px] font-display text-muted-foreground">
              +{members.length - 3}
            </div>
          )}
        </div>
        <div className="text-left hidden sm:block">
          <p className="text-xs font-display font-semibold text-foreground leading-tight max-w-[80px] truncate">{activeMember.name}</p>
          <p className="text-[10px] text-muted-foreground">{activeMember.relation}</p>
        </div>
        <ChevronDown className={cn("w-3.5 h-3.5 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute top-full right-0 mt-2 w-80 glass-card-premium p-2 z-50 animate-fade-in-up shadow-xl">
          <p className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-neurora-gold font-display font-semibold flex items-center gap-1">
            <Crown className="w-3 h-3" /> Family Nexus
          </p>

          <div className="space-y-1 max-h-[300px] overflow-y-auto custom-scrollbar">
            {members.map((m) => (
              <div key={m.id} className="group relative">
                <button
                  onClick={() => { setActiveMember(m); setOpen(false); }}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all",
                    m.id === activeMember.id ? "bg-neurora-gold/10" : "hover:bg-secondary/50",
                    m.status === 'pending' && "opacity-70"
                  )}
                >
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-display font-bold text-white shrink-0 relative"
                    style={{ backgroundColor: m.accentColor }}
                  >
                    {m.avatar}
                    {m.status === 'pending' && (
                      <span className="absolute -top-1 -right-1 w-3 h-3 bg-neurora-amber rounded-full border border-background flex items-center justify-center">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neurora-amber opacity-75"></span>
                      </span>
                    )}
                  </div>
                  <div className="text-left flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-display font-semibold text-foreground truncate">{m.name}</p>
                      {m.status === 'pending' && <span className="text-[10px] bg-neurora-amber/20 text-neurora-amber px-1.5 rounded-full">Pending</span>}
                    </div>
                    <p className="text-xs text-muted-foreground">{m.relation}</p>
                  </div>
                  {m.id === activeMember.id && <Check className="w-4 h-4 text-neurora-gold shrink-0" />}
                </button>

                {!m.isPrimary && (
                  <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    {m.status === 'pending' && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-neurora-mint hover:text-neurora-mint hover:bg-neurora-mint/10"
                        onClick={(e) => handleSimulateAccept(e, m.id)}
                        title="Simulate Accept (Dev Only)"
                      >
                        <Check className="w-3.5 h-3.5" />
                      </Button>
                    )}
                    <button
                      onClick={(e) => handleRemoveMember(e, m.id, m.name)}
                      className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-md"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="border-t border-border/30 mt-2 pt-2">
            {!showAddForm ? (
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-xs h-9 gap-2 text-muted-foreground hover:text-primary"
                onClick={() => setShowAddForm(true)}
              >
                <Plus className="w-3.5 h-3.5" /> Add Family Member
              </Button>
            ) : (
              <div className="p-2 space-y-3 bg-secondary/20 rounded-lg animate-in fade-in slide-in-from-top-1">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold">Invite Member</p>
                  <button onClick={() => setShowAddForm(false)} className="text-muted-foreground hover:text-foreground"><X className="w-3 h-3" /></button>
                </div>
                <div className="space-y-2">
                  <Input
                    placeholder="Name"
                    className="h-8 text-xs"
                    value={newMemberName}
                    onChange={(e) => setNewMemberName(e.target.value)}
                  />
                  <Input
                    placeholder="Relation (e.g. Son)"
                    className="h-8 text-xs"
                    value={newMemberRelation}
                    onChange={(e) => setNewMemberRelation(e.target.value)}
                  />
                  <div className="relative">
                    <Mail className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground" />
                    <Input
                      placeholder="Email (Invite)"
                      className="h-8 text-xs pl-8"
                      value={newMemberEmail}
                      onChange={(e) => setNewMemberEmail(e.target.value)}
                    />
                  </div>
                  <Button size="sm" className="w-full h-7 text-xs gap-1 glow-indigo" onClick={handleAddMember}>
                    <UserPlus className="w-3 h-3" /> {newMemberEmail ? "Send Invite" : "Add Profile"}
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
