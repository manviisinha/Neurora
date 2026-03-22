
import { useEffect, useState } from 'react';

export function HeroVisualization() {
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    return (
        <div className="relative w-full h-[500px] flex items-center justify-center perspective-1000">
            {/* Background Glow */}
            <div className="absolute inset-0 bg-neurora-indigo/5 blur-3xl rounded-full scale-75 animate-pulse" />

            <svg viewBox="0 0 400 400" className="w-full h-full max-w-[500px] visible overflow-visible">
                <defs>
                    <radialGradient id="grad-core" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                        <stop offset="0%" stopColor="hsl(var(--neurora-indigo))" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="hsl(var(--neurora-indigo))" stopOpacity="0.2" />
                    </radialGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Connecting Lines (Synapses) */}
                <g className="opacity-40">
                    {/* Center to Orbit 1 */}
                    <line x1="200" y1="200" x2="200" y2="100" stroke="hsl(var(--neurora-indigo))" strokeWidth="1" strokeDasharray="4 4" className="animate-pulse" />
                    <line x1="200" y1="200" x2="286" y2="150" stroke="hsl(var(--neurora-indigo))" strokeWidth="1" strokeDasharray="4 4" className="animate-pulse delay-75" />
                    <line x1="200" y1="200" x2="286" y2="250" stroke="hsl(var(--neurora-indigo))" strokeWidth="1" strokeDasharray="4 4" className="animate-pulse delay-150" />
                    <line x1="200" y1="200" x2="200" y2="300" stroke="hsl(var(--neurora-indigo))" strokeWidth="1" strokeDasharray="4 4" className="animate-pulse delay-200" />
                    <line x1="200" y1="200" x2="114" y2="250" stroke="hsl(var(--neurora-indigo))" strokeWidth="1" strokeDasharray="4 4" className="animate-pulse delay-300" />
                    <line x1="200" y1="200" x2="114" y2="150" stroke="hsl(var(--neurora-indigo))" strokeWidth="1" strokeDasharray="4 4" className="animate-pulse delay-400" />
                </g>

                {/* Orbital Rings */}
                <circle cx="200" cy="200" r="100" fill="none" stroke="hsl(var(--neurora-indigo))" strokeWidth="0.5" strokeOpacity="0.3" className="animate-[spin_10s_linear_infinite]" style={{ transformOrigin: 'center' }} />
                <circle cx="200" cy="200" r="140" fill="none" stroke="hsl(var(--neurora-gold))" strokeWidth="0.5" strokeOpacity="0.2" className="animate-[spin_15s_linear_infinite_reverse]" style={{ transformOrigin: 'center' }} />

                {/* Central Core */}
                <g className="animate-brain-pulse">
                    <circle cx="200" cy="200" r="30" fill="url(#grad-core)" filter="url(#glow)" />
                    <circle cx="200" cy="200" r="15" fill="hsl(var(--neurora-indigo))" opacity="0.8" />
                </g>

                {/* Floating Satellites (Data Points) */}
                {/* Top */}
                <g className="animate-[bounce_3s_infinite]">
                    <circle cx="200" cy="100" r="8" fill="hsl(var(--neurora-mint))" opacity="0.8" filter="url(#glow)" />
                    <text x="215" y="105" fill="hsl(var(--muted-foreground))" fontSize="10" className="opacity-70 font-display">Analysis</text>
                </g>

                {/* Top Right */}
                <g className="animate-[bounce_4s_infinite]" style={{ animationDelay: '1s' }}>
                    <circle cx="286" cy="150" r="6" fill="hsl(var(--neurora-gold))" opacity="0.8" filter="url(#glow)" />
                </g>

                {/* Bottom Right */}
                <g className="animate-[bounce_3.5s_infinite]" style={{ animationDelay: '0.5s' }}>
                    <circle cx="286" cy="250" r="7" fill="hsl(var(--neurora-amethyst))" opacity="0.8" filter="url(#glow)" />
                    <text x="300" y="255" fill="hsl(var(--muted-foreground))" fontSize="10" className="opacity-70 font-display">Genetics</text>
                </g>

                {/* Bottom */}
                <g className="animate-[bounce_4.5s_infinite]" style={{ animationDelay: '1.5s' }}>
                    <circle cx="200" cy="300" r="5" fill="hsl(var(--neurora-indigo))" opacity="0.6" />
                </g>

                {/* Bottom Left */}
                <g className="animate-[bounce_3.2s_infinite]" style={{ animationDelay: '0.8s' }}>
                    <circle cx="114" cy="250" r="8" fill="hsl(var(--neurora-mint))" opacity="0.8" filter="url(#glow)" />
                </g>

                {/* Top Left */}
                <g className="animate-[bounce_3.8s_infinite]" style={{ animationDelay: '1.2s' }}>
                    <circle cx="114" cy="150" r="6" fill="hsl(var(--neurora-gold))" opacity="0.8" filter="url(#glow)" />
                    <text x="50" y="155" fill="hsl(var(--muted-foreground))" fontSize="10" className="opacity-70 font-display">Vitals</text>
                </g>

                {/* Moving Data Packets along lines */}
                <circle r="3" fill="hsl(var(--neurora-indigo))">
                    <animateMotion dur="2s" repeatCount="indefinite" path="M200,200 L200,100" />
                </circle>
                <circle r="3" fill="hsl(var(--neurora-gold))">
                    <animateMotion dur="3s" repeatCount="indefinite" path="M286,150 L200,200" />
                </circle>
                <circle r="3" fill="hsl(var(--neurora-mint))">
                    <animateMotion dur="2.5s" repeatCount="indefinite" path="M114,250 L200,200" />
                </circle>

            </svg>
        </div>
    );
}
