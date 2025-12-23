import { FileText, X } from 'lucide-react';

interface SummaryTopic {
  time?: string;
  topic: string;
  detail: string;
}

interface SummaryData {
  duration: string;
  topicsDiscussed: SummaryTopic[];
  sentiment: string;
  nextSteps: string[];
}

interface SummarizePanelProps {
  onClose: () => void;
  isKnownPerson: boolean;
  summaryData?: SummaryData;
}

export function SummarizePanel({ onClose, isKnownPerson, summaryData }: SummarizePanelProps) {
  const knownPersonSummary: SummaryData = {
    duration: "3 minutes",
    topicsDiscussed: [
      {
        time: "0:15",
        topic: "Initial greeting and recognition",
        detail: "Met Sarah Chen, reconnected since June coffee meetup"
      },
      {
        time: "0:45",
        topic: "Recent design work",
        detail: "Sarah shared updates on new AR interface project for mobile apps"
      },
      {
        time: "1:30",
        topic: "Photography discussion",
        detail: "Talked about landscape photography techniques and favorite spots"
      },
      {
        time: "2:15",
        topic: "Weekend hiking plans",
        detail: "Discussed trails near the bay area, considering joint hike next month"
      },
      {
        time: "2:50",
        topic: "Follow-up plans",
        detail: "Agreed to share photography exhibition details and schedule coffee"
      }
    ],
    sentiment: "Positive and engaging",
    nextSteps: [
      "Share photography exhibition info",
      "Send hiking trail recommendations",
      "Schedule coffee meetup next week"
    ]
  };

  const newPersonSummary: SummaryData = {
    duration: "2 minutes",
    topicsDiscussed: [
      {
        time: "0:10",
        topic: "Initial introduction",
        detail: "Met Alex Martinez at the tech conference, exchanged names"
      },
      {
        time: "0:40",
        topic: "Conference discussion",
        detail: "Talked about the keynote presentation on AI innovations"
      },
      {
        time: "1:20",
        topic: "Professional background",
        detail: "Alex works in startup tech, discussed their recent product launch"
      },
      {
        time: "1:50",
        topic: "Networking",
        detail: "Exchanged contact information and LinkedIn profiles"
      }
    ],
    sentiment: "Friendly and professional",
    nextSteps: [
      "Connect on LinkedIn",
      "Share conference notes",
      "Follow up about startup collaboration"
    ]
  };

  const data = summaryData ?? (isKnownPerson ? knownPersonSummary : newPersonSummary);

  return (
    <div className="absolute top-20 left-8 z-40 animate-fade-in">
      <div className="w-[420px] bg-blue-500/25 backdrop-blur-xl border border-blue-400/40 rounded-2xl p-5 shadow-2xl max-h-[calc(100vh-160px)] overflow-y-auto scrollbar-hide">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-full bg-purple-400/20 flex items-center justify-center">
              <FileText className="w-5 h-5 text-purple-300" />
            </div>
            <div>
              <h3 className="text-white">Conversation Summary</h3>
              <p className="text-white/60 text-sm">{data.duration} conversation</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-all"
            aria-label="Close summary"
          >
            <X className="w-4 h-4 text-white/70" />
          </button>
        </div>

        {/* Divider */}
        <div className="h-px bg-white/20 mb-4" />

        {/* Topics Discussed */}
        <div className="mb-4">
          <p className="text-white/60 text-xs mb-3">Topics Discussed</p>
          <div className="space-y-3">
            {data.topicsDiscussed.map((item, index) => (
              <div key={index} className="relative pl-6">
                {/* Timeline indicator */}
                <div className="absolute left-0 top-0">
                  <div className="w-4 h-4 rounded-full bg-blue-400/30 border-2 border-blue-400 flex items-center justify-center">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-300" />
                  </div>
                  {index < data.topicsDiscussed.length - 1 && (
                    <div className="absolute left-1/2 top-4 w-px h-8 bg-white/20 -ml-px" />
                  )}
                </div>
                
                {/* Content */}
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-white">{item.topic}</p>
                    {item.time && <span className="text-blue-300 text-xs">{item.time}</span>}
                  </div>
                  <p className="text-white/70 text-sm">{item.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="h-px bg-white/20 my-4" />

        {/* Next Steps */}
        <div className="mb-4">
          <p className="text-white/60 text-xs mb-2">Next Steps</p>
          <div className="space-y-2">
            {data.nextSteps.map((step, index) => (
              <div key={index} className="flex gap-2 items-start">
                <div className="w-5 h-5 rounded bg-green-400/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-green-300 text-xs">{index + 1}</span>
                </div>
                <p className="text-white text-sm">{step}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Sentiment */}
        <div className="pt-3 border-t border-white/20">
          <div className="flex items-center justify-between">
            <p className="text-white/60 text-xs">Overall Sentiment</p>
            <p className="text-white text-sm">{data.sentiment}</p>
          </div>
        </div>

        {/* AI indicator */}
        <div className="mt-3 pt-3 border-t border-white/20">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
            <span className="text-white/60 text-xs">AI-powered conversation analysis</span>
          </div>
        </div>
      </div>
    </div>
  );
}
