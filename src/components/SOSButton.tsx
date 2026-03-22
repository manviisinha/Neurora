import { Phone } from "lucide-react";

export function SOSButton() {
  return (
    <button
      onClick={() => {
        if (window.confirm("Call Emergency Services (112)?")) {
          window.location.href = "tel:112";
        }
      }}
      className="fixed bottom-6 left-6 z-50 w-14 h-14 rounded-full bg-destructive flex items-center justify-center animate-pulse-glow transition-transform hover:scale-110 shadow-lg shadow-destructive/40"
      title="Emergency SOS"
    >
      <Phone className="w-6 h-6 text-destructive-foreground animate-pulse" />
    </button>
  );
}
