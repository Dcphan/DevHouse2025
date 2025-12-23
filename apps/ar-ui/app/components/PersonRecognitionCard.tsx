import { User, Calendar, MessageCircle, Heart, Sparkles } from 'lucide-react';

interface PersonData {
  name: string;
  firstMet: string;
  relationship: string;
  lastTopic: string | null;
  interests: string[];
}

interface PersonRecognitionCardProps {
  person: PersonData;
  isNewPerson?: boolean;
}

export function PersonRecognitionCard({ person, isNewPerson = false }: PersonRecognitionCardProps) {
  return (
    <div className="absolute top-20 right-8 z-40 animate-fade-in">
      <div className="w-80 bg-blue-500/25 backdrop-blur-xl border border-blue-400/40 rounded-2xl p-5 shadow-2xl">
        {/* Header with avatar */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center">
            <User className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-white font-semibold">{person.name}</h3>
            <p className="text-white/70 text-sm">{person.relationship}</p>
          </div>
          {isNewPerson && (
            <div className="px-2 py-1 bg-green-400/30 rounded-full border border-green-400/40">
              <span className="text-green-200 text-xs">New</span>
            </div>
          )}
        </div>

        {/* Divider */}
        <div className="h-px bg-white/20 mb-4" />

        {/* Info sections */}
        <div className="space-y-3">
          <div className="flex items-start gap-2">
            <Calendar className="w-4 h-4 text-blue-300 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-white/60 text-xs">
                {isNewPerson ? 'Meeting for the first time' : 'First met'}
              </p>
              <p className="text-white text-sm">{person.firstMet}</p>
            </div>
          </div>

          {!isNewPerson && person.lastTopic && (
            <div className="flex items-start gap-2">
              <MessageCircle className="w-4 h-4 text-green-300 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-white/60 text-xs">Last conversation</p>
                <p className="text-white text-sm">{person.lastTopic}</p>
              </div>
            </div>
          )}

          {isNewPerson && (
            <div className="flex items-start gap-2">
              <Sparkles className="w-4 h-4 text-yellow-300 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-white/60 text-xs">Name captured</p>
                <p className="text-white text-sm">Creating profile & suggestions</p>
              </div>
            </div>
          )}

          <div className="flex items-start gap-2">
            <Heart className="w-4 h-4 text-pink-300 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-white/60 text-xs">
                {isNewPerson ? 'Detected interests' : 'Shared interests'}
              </p>
              <div className="flex flex-wrap gap-1.5 mt-1">
                {person.interests.map((interest, index) => (
                  <span
                    key={index}
                    className="px-2 py-0.5 bg-white/20 rounded-full text-white text-xs"
                  >
                    {interest}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Recognition indicator */}
        <div className="mt-4 pt-3 border-t border-white/20">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-white/60 text-xs">
              {isNewPerson ? 'Identified via Meta AI' : 'Recognized via Meta AI'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
