import { Lightbulb } from 'lucide-react';

interface ConversationSuggestionsProps {
  suggestions: string[];
}

export function ConversationSuggestions({ suggestions }: ConversationSuggestionsProps) {
  if (suggestions.length === 0) return null;

  return (
    <div className="absolute bottom-32 left-8 z-40 animate-slide-up">
      <div className="w-96 bg-blue-500/25 backdrop-blur-xl border border-blue-400/40 rounded-2xl p-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <div className="w-8 h-8 rounded-full bg-yellow-400/20 flex items-center justify-center">
            <Lightbulb className="w-4 h-4 text-yellow-300" />
          </div>
          <span className="text-white/80 text-sm">Conversation Ideas</span>
        </div>

        {/* Suggestions */}
        <div className="space-y-2">
          {suggestions.map((suggestion, index) => (
            <div
              key={index}
              className="p-3 bg-white/10 rounded-xl border border-white/10 hover:bg-white/20 transition-all cursor-pointer group"
            >
              <p className="text-white text-sm group-hover:text-white/90">
                {suggestion}
              </p>
            </div>
          ))}
        </div>

        {/* Subtle hint */}
        <p className="text-white/40 text-xs mt-3 text-center">
          Tap to use in conversation
        </p>
      </div>
    </div>
  );
}
