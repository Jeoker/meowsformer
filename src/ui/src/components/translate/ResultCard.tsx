import { useEffect } from "react";
import { useAudioPreview } from "../../hooks/useAudioPreview";
import type { TaggedSampleInfo } from "../../types/api";

interface ResultCardProps {
  transcription: string;
  sample: TaggedSampleInfo | null;
  audioBase64: string | null;
  reasoning: string;
}

export default function ResultCard({ transcription, sample, audioBase64, reasoning }: ResultCardProps) {
  const audio = useAudioPreview();

  useEffect(() => {
    if (audioBase64) {
      audio.loadBase64(audioBase64);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioBase64]);

  const isPlaying = audio.state === "playing";

  function toggle() {
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4">
      {/* Transcription */}
      <div>
        <p className="text-xs uppercase tracking-widest text-gray-500 mb-1">You said</p>
        <p className="text-gray-200 italic">"{transcription}"</p>
      </div>

      {/* Tags */}
      {sample && (
        <div className="flex flex-wrap gap-2">
          {sample.tags.emotion?.map((t) => (
            <TagBadge key={t} label={t} color="purple" />
          ))}
          {sample.tags.intent?.map((t) => (
            <TagBadge key={t} label={t} color="blue" />
          ))}
          {sample.tags.acoustic?.slice(0, 3).map((t) => (
            <TagBadge key={t} label={t} color="green" />
          ))}
        </div>
      )}

      {/* Reasoning */}
      {reasoning && (
        <p className="text-sm text-gray-400 leading-relaxed">{reasoning}</p>
      )}

      {/* Audio player */}
      {audioBase64 && (
        <button
          onClick={toggle}
          disabled={audio.state === "loading"}
          className="flex items-center gap-3 bg-meow-600/20 hover:bg-meow-600/30 border border-meow-700 rounded-xl px-4 py-3 w-full transition group disabled:opacity-50"
        >
          <span className="text-2xl">{isPlaying ? "⏸" : "▶️"}</span>
          <div className="text-left">
            <p className="text-sm font-medium text-meow-300 group-hover:text-meow-200">
              {isPlaying ? "Pause meow" : "Play meow"}
            </p>
            {sample && (
              <p className="text-xs text-gray-500">
                Match score: {(sample.match_score * 100).toFixed(0)}%
              </p>
            )}
          </div>
        </button>
      )}
    </div>
  );
}

function TagBadge({ label, color }: { label: string; color: "purple" | "blue" | "green" }) {
  const colorMap = {
    purple: "bg-purple-900/40 text-purple-300 border-purple-800",
    blue: "bg-blue-900/40 text-blue-300 border-blue-800",
    green: "bg-green-900/40 text-green-300 border-green-800",
  };
  return (
    <span className={`text-xs px-2.5 py-1 rounded-full border font-medium ${colorMap[color]}`}>
      {label}
    </span>
  );
}
