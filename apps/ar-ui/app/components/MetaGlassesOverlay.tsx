import { Wifi, Battery, Clock } from 'lucide-react';
import { useState, useEffect } from 'react';

export function MetaGlassesOverlay() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit',
      hour12: true 
    });
  };

  return (
    <>
      {/* Top status bar */}
      <div className="absolute top-0 left-0 right-0 z-30 p-6">
        <div className="flex items-center justify-between">
          {/* Left side - Meta logo indicator */}
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            <span className="text-white/60 text-sm">Meta View</span>
          </div>

          {/* Right side - Status icons */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <Clock className="w-4 h-4 text-white/70" />
              <span className="text-white/70 text-sm">{formatTime(time)}</span>
            </div>
            <Wifi className="w-4 h-4 text-white/70" />
            <div className="flex items-center gap-1">
              <Battery className="w-4 h-4 text-white/70" />
              <span className="text-white/70 text-xs">78%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Subtle corner frame indicators - simulating AR frame */}
      <div className="absolute top-8 left-8 w-12 h-12 border-l-2 border-t-2 border-white/20 z-20" />
      <div className="absolute top-8 right-8 w-12 h-12 border-r-2 border-t-2 border-white/20 z-20" />
      <div className="absolute bottom-8 left-8 w-12 h-12 border-l-2 border-b-2 border-white/20 z-20" />
      <div className="absolute bottom-8 right-8 w-12 h-12 border-r-2 border-b-2 border-white/20 z-20" />

      {/* Center focus indicator (subtle) */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-20">
        <div className="w-1 h-1 rounded-full bg-white/30" />
      </div>
    </>
  );
}
