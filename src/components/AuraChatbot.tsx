import { useState, useEffect, useRef } from "react";
import { MessageCircle, X, Send, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { auth, db } from "@/lib/firebase";
import { useFamily } from "@/contexts/FamilyContext";
import { collection, query, orderBy, limit, getDocs, onSnapshot } from "firebase/firestore";
import { DRUG_DATABASE } from "@/services/medicationKnowledgeBase";

type Message = { role: "user" | "assistant"; text: string };

export function AuraChatbot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Hello! I'm Aura, your AI medical assistant. I can help answer questions about your medicines, lab reports, or potential conflicts." },
  ]);
  const [input, setInput] = useState("");
  const { activeMember } = useFamily();
  const [meds, setMeds] = useState<any[]>([]);
  const [reports, setReports] = useState<any[]>([]);

  // Ref for auto-scrolling
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch data on mount/activeMember change to be ready for queries
  useEffect(() => {
    if (!auth.currentUser || !activeMember) return;

    // 1. Fetch Medications
    const medPath = activeMember.isPrimary
      ? `users/${auth.currentUser.uid}/medications`
      : `users/${auth.currentUser.uid}/family_members/${activeMember.id}/medications`;

    const unsubMeds = onSnapshot(query(collection(db, medPath)), (snap) => {
      setMeds(snap.docs.map(d => d.data()));
    });

    // 2. Fetch Lab Reports
    const labPath = activeMember.isPrimary
      ? `users/${auth.currentUser.uid}/lab_reports`
      : `users/${auth.currentUser.uid}/family_members/${activeMember.id}/lab_reports`;

    const unsubLabs = onSnapshot(query(collection(db, labPath), orderBy("date", "desc"), limit(5)), (snap) => {
      setReports(snap.docs.map(d => ({ ...d.data(), date: d.data().date?.toDate() })));
    });

    return () => {
      unsubMeds();
      unsubLabs();
    };
  }, [activeMember]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const generateResponse = (query: string) => {
    const lowerQ = query.toLowerCase();

    // 1. Conflict Check
    if (lowerQ.includes("conflict") || lowerQ.includes("interaction") || lowerQ.includes("safe")) {
      const medNames = meds.map(m => m.name.toLowerCase());
      const conflicts: string[] = [];

      meds.forEach(med => {
        const info = Object.values(DRUG_DATABASE).find(d => d.name.toLowerCase() === med.name.toLowerCase());
        if (info) {
          info.interactions.forEach(int => {
            if (medNames.includes(int.drug.toLowerCase())) {
              conflicts.push(`${info.name} + ${int.drug}: ${int.description}`);
            }
          });
        }
      });

      if (conflicts.length > 0) {
        // Deduplicate simplistic strings
        const unique = Array.from(new Set(conflicts));
        return `I found ${unique.length} potential interaction(s) in your current medications:\n• ${unique.join("\n• ")}`;
      } else {
        return "I've analyzed your current medication list, and I don't see any known major drug interactions based on my database. However, always consult your doctor.";
      }
    }

    // 2. Lab Reports
    if (lowerQ.includes("glucose") || lowerQ.includes("sugar") || lowerQ.includes("a1c")) {
      const relevant = reports.flatMap(r => r.tests?.filter((t: any) =>
        t.name.toLowerCase().includes("glucose") ||
        t.name.toLowerCase().includes("sugar") ||
        t.name.toLowerCase().includes("a1c")
      ) || []);

      if (relevant.length > 0) {
        const latest = relevant[0];
        return `Your latest ${latest.name} reading was ${latest.value} ${latest.unit}. Check the Lab Analytics page for a full trend analysis.`;
      }
      return "I couldn't find any recent glucose or A1C records in your uploaded reports.";
    }

    if (lowerQ.includes("cholesterol") || lowerQ.includes("lipid")) {
      const relevant = reports.flatMap(r => r.tests?.filter((t: any) => t.name.toLowerCase().includes("cholesterol")) || []);
      if (relevant.length > 0) {
        const latest = relevant[0];
        return `Your most recent cholesterol level on file is ${latest.value} ${latest.unit}.`;
      }
      return "No recent cholesterol data found.";
    }

    if (lowerQ.includes("report") || lowerQ.includes("lab")) {
      if (reports.length > 0) {
        return `You have ${reports.length} recent lab report(s) uploaded. The latest one is from ${reports[0].date?.toLocaleDateString() || 'unknown date'}.`;
      }
      return "You haven't uploaded any lab reports yet. Upload one to get started!";
    }

    // 3. Medications
    if (lowerQ.includes("medicine") || lowerQ.includes("drug") || lowerQ.includes("pill") || lowerQ.includes("taking")) {
      if (meds.length > 0) {
        const names = meds.map(m => m.name).join(", ");
        return `You are currently tracking ${meds.length} medication(s): ${names}.`;
      }
      return "Your medication info is empty. Try scanning a prescription!";
    }

    // 4. Default / Greetings
    if (lowerQ.includes("hello") || lowerQ.includes("hi")) return `Hello ${activeMember?.name || 'there'}! How can I help with your health data today?`;
    if (lowerQ.includes("thank")) return "You're welcome! Stay healthy.";
    if (lowerQ.includes("emergency") || lowerQ.includes("sos") || lowerQ.includes("help")) return "If you are in a medical emergency, please press the red SOS button on your screen immediately or dial 112.";

    return "I'm still learning constraints of your specific health data. Try asking about your 'medications', 'conflicts', or 'latest glucose levels'.";
  };

  const handleSend = () => {
    if (!input.trim()) return;

    const userMsg = input;
    setMessages((prev) => [...prev, { role: "user", text: userMsg }]);
    setInput("");

    // Simulate thinking delay
    setTimeout(() => {
      const response = generateResponse(userMsg);
      setMessages((prev) => [...prev, { role: "assistant", text: response }]);
    }, 600);
  };

  return (
    <>
      {/* Chat bubble */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-primary flex items-center justify-center glow-indigo transition-transform hover:scale-110 overflow-hidden"
        >
          {/* Neural wave effect */}
          <div className="absolute inset-0 opacity-30">
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary-foreground/20 to-transparent animate-neural-wave" />
          </div>
          <MessageCircle className="w-6 h-6 text-primary-foreground relative z-10" />
        </button>
      )}

      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-6 right-6 z-50 w-[360px] h-[500px] glass-card flex flex-col animate-fade-in-up overflow-hidden shadow-2xl border-primary/20">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border/30 bg-secondary/40 backdrop-blur-md">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-full bg-primary/20 border border-primary/10 flex items-center justify-center relative">
                <div className="absolute inset-0 rounded-full animate-pulse bg-primary/5"></div>
                <MessageCircle className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-sm font-display font-semibold text-foreground flex items-center gap-1">
                  Aura AI <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">BETA</span>
                </p>
                <div className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-neurora-mint animate-pulse"></span>
                  <p className="text-[10px] text-muted-foreground">Connected to {activeMember?.name}'s Health Data</p>
                </div>
              </div>
            </div>
            <button onClick={() => setOpen(false)} className="w-8 h-8 rounded-full bg-secondary/50 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors hover:bg-destructive/10 hover:text-destructive">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-auto p-4 space-y-4 bg-gradient-to-b from-background/50 to-secondary/10">
            {messages.map((msg, i) => (
              <div key={i} className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}>
                {msg.role === "assistant" && (
                  <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center mr-2 mt-1 shrink-0">
                    <MessageCircle className="w-3 h-3 text-primary" />
                  </div>
                )}
                <div
                  className={cn(
                    "max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm",
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground rounded-tr-sm"
                      : "bg-secondary/80 text-foreground border border-border/50 rounded-tl-sm whitespace-pre-wrap"
                  )}
                >
                  {msg.text}
                </div>
                {msg.role === "user" && (
                  <div className="w-6 h-6 rounded-full bg-secondary/50 flex items-center justify-center ml-2 mt-1 shrink-0">
                    {activeMember?.avatar ? (
                      <span className="text-[8px] font-bold">{activeMember.avatar}</span>
                    ) : (
                      <User className="w-3 h-3 opacity-50" />
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-3 border-t border-border/30 bg-background/80 backdrop-blur-md">
            <div className="flex items-center gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder="Ask about conflicts, labs, or meds..."
                className="flex-1 h-10 rounded-xl bg-secondary/50 border border-border/30 px-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 transition-all"
                autoFocus
              />
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center text-primary-foreground hover:bg-primary/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95"
              >
                <Send className="w-4 h-4 ml-0.5" />
              </button>
            </div>
            <p className="text-[9px] text-center text-muted-foreground mt-2 opacity-60">
              Aura can make mistakes. Verify important medical info.
            </p>
          </div>
        </div>
      )}
    </>
  );
}
