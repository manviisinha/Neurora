export function MoleculeVisualization() {
  return (
    <div className="relative w-full h-48 flex items-center justify-center">
      <svg viewBox="0 0 200 200" className="w-40 h-40 opacity-80">
        {/* Central atom */}
        <circle cx="100" cy="100" r="16" fill="hsl(260, 58%, 53%)" opacity="0.9" className="animate-brain-pulse" />
        <circle cx="100" cy="100" r="22" fill="none" stroke="hsl(260, 58%, 53%)" strokeWidth="1" opacity="0.4" />

        {/* Bonds and outer atoms */}
        <line x1="100" y1="100" x2="55" y2="55" stroke="hsl(220, 20%, 40%)" strokeWidth="2" />
        <circle cx="55" cy="55" r="10" fill="hsl(157, 100%, 50%)" opacity="0.7" />

        <line x1="100" y1="100" x2="150" y2="50" stroke="hsl(220, 20%, 40%)" strokeWidth="2" />
        <circle cx="150" cy="50" r="12" fill="hsl(42, 100%, 50%)" opacity="0.7" />

        <line x1="100" y1="100" x2="145" y2="145" stroke="hsl(220, 20%, 40%)" strokeWidth="2" />
        <circle cx="145" cy="145" r="10" fill="hsl(260, 58%, 53%)" opacity="0.6" />

        <line x1="100" y1="100" x2="50" y2="140" stroke="hsl(220, 20%, 40%)" strokeWidth="2" />
        <circle cx="50" cy="140" r="8" fill="hsl(157, 100%, 50%)" opacity="0.5" />

        <line x1="100" y1="100" x2="100" y2="40" stroke="hsl(220, 20%, 40%)" strokeWidth="2" />
        <circle cx="100" cy="40" r="9" fill="hsl(0, 72%, 51%)" opacity="0.6" />

        {/* Orbital rings */}
        <ellipse cx="100" cy="100" rx="60" ry="25" fill="none" stroke="hsl(260, 58%, 53%)" strokeWidth="0.5" opacity="0.25" transform="rotate(30 100 100)" />
        <ellipse cx="100" cy="100" rx="60" ry="25" fill="none" stroke="hsl(260, 58%, 53%)" strokeWidth="0.5" opacity="0.25" transform="rotate(-30 100 100)" />
      </svg>
    </div>
  );
}
