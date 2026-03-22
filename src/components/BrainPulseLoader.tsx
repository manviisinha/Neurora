import { Brain } from "lucide-react";

export function BrainPulseLoader({ text = "Processing..." }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <Brain className="w-10 h-10 text-primary animate-brain-pulse" />
      <p className="text-sm text-muted-foreground font-display">{text}</p>
    </div>
  );
}
