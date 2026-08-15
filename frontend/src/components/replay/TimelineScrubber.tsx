import React, { useState } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Clock,
  Sliders,
  GitCompare,
} from "lucide-react";
import { ClusterSnapshot, TimelineState } from "../../types";

interface TimelineScrubberProps {
  timelineState: TimelineState;
  cluster: ClusterSnapshot | null;
  onSeekTick: (tick: number) => void;
  onTogglePlay: () => void;
  onSetSpeed: (speed: number) => void;
  onSetDiffTick?: (tick: number | null) => void;
  onClose?: () => void;
}

export const TimelineScrubber: React.FC<TimelineScrubberProps> = ({
  timelineState,
  cluster,
  onSeekTick,
  onTogglePlay,
  onSetSpeed,
  onSetDiffTick,
}) => {
  const [showDiffPicker, setShowDiffPicker] = useState(false);

  const currentTick = cluster ? cluster.tick : timelineState.currentTick;
  const maxTick = Math.max(200, timelineState.maxTick, currentTick);

  const handleScrubChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTick = parseInt(e.target.value, 10);
    onSeekTick(targetTick);
  };

  return (
    <div className="absolute bottom-4 left-4 right-4 z-40 bg-slate-950/90 backdrop-blur-xl border border-purple-900/40 px-5 py-3 rounded-2xl shadow-2xl text-slate-100 flex flex-col space-y-2.5 text-xs">
      {/* Upper Control Strip */}
      <div className="flex items-center justify-between">
        {/* Title & Live Status */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 text-purple-400 font-bold">
            <Clock className="w-4 h-4 text-purple-400" />
            <span>TIME-TRAVEL REPLAY SCRUBBER</span>
          </div>
          <span className="font-mono text-xs bg-purple-950 text-purple-300 border border-purple-800/60 px-2.5 py-0.5 rounded-full font-bold">
            TICK {currentTick} / {maxTick}
          </span>
        </div>

        {/* Playback Button Group */}
        <div className="flex items-center space-x-2">
          {/* Step Back */}
          <button
            onClick={() => onSeekTick(Math.max(0, currentTick - 1))}
            className="p-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-slate-300 transition-colors"
            title="Step Back 1 Tick"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          {/* Play / Pause */}
          <button
            onClick={onTogglePlay}
            className={`px-3 py-1.5 rounded-xl border font-bold text-xs flex items-center space-x-1.5 transition-all ${
              timelineState.isPlaying
                ? "bg-amber-500/20 border-amber-500/40 text-amber-300 hover:bg-amber-500/30"
                : "bg-purple-600/30 border-purple-500/50 text-purple-200 hover:bg-purple-600/40"
            }`}
          >
            {timelineState.isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{timelineState.isPlaying ? "PAUSE" : "REPLAY"}</span>
          </button>

          {/* Step Forward */}
          <button
            onClick={() => onSeekTick(Math.min(maxTick, currentTick + 1))}
            className="p-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-slate-300 transition-colors"
            title="Step Forward 1 Tick"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          {/* Speed Selector */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-xl p-0.5 font-mono text-[11px]">
            {[0.5, 1.0, 2.0, 5.0].map((speed) => (
              <button
                key={speed}
                onClick={() => onSetSpeed(speed)}
                className={`px-2 py-0.5 rounded-lg font-bold transition-colors ${
                  timelineState.playbackSpeed === speed
                    ? "bg-purple-600 text-white shadow"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>

          {/* 3D Delta Diff Toggle */}
          <button
            onClick={() => {
              const nextVal = showDiffPicker ? null : Math.max(0, currentTick - 10);
              setShowDiffPicker(!showDiffPicker);
              if (onSetDiffTick) onSetDiffTick(nextVal);
            }}
            className={`px-2.5 py-1 rounded-xl border text-xs font-medium flex items-center space-x-1 transition-all ${
              showDiffPicker
                ? "bg-purple-900/60 border-purple-500 text-purple-200"
                : "bg-slate-900 border-slate-800 text-slate-400 hover:text-purple-300"
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            <span>3D DELTA DIFF</span>
          </button>
        </div>
      </div>

      {/* Main Scrubber Slider */}
      <div className="flex items-center space-x-3">
        <span className="font-mono text-[10px] text-slate-400">0</span>
        <input
          type="range"
          min="0"
          max={maxTick}
          value={currentTick}
          onChange={handleScrubChange}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400 transition-all"
        />
        <span className="font-mono text-[10px] text-slate-400">{maxTick}</span>
      </div>
    </div>
  );
};
