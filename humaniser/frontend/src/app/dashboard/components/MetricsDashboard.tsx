import React from "react";
import { 
  BarChart3, 
  TrendingDown, 
  Zap, 
  Search, 
  AlertCircle,
  ArrowDownRight,
  Clock
} from "lucide-react";
import { clsx } from "clsx";
import { HumanizationMetrics } from "../api";

interface MetricsDashboardProps {
  metrics?: HumanizationMetrics;
}

const MetricCard = ({ 
  label, 
  baseline, 
  humanized, 
  inverse = false 
}: { 
  label: string; 
  baseline: number; 
  humanized: number; 
  inverse?: boolean; 
}) => {
  const delta = humanized - baseline;
  // If inverse=true, a decrease is good (like AI score). If false, an increase is good (like Burstiness/Perplexity usually).
  const isGood = inverse ? delta < 0 : delta > 0;
  const percentage = baseline > 0 ? (Math.abs(delta) / baseline) * 100 : 0;

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm transition-all hover:shadow-md">
      <div className="flex justify-between items-start mb-4">
        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{label}</span>
        {delta !== 0 && (
          <div className={clsx(
            "flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-black uppercase",
            isGood ? "bg-emerald-50 text-emerald-600" : "bg-rose-50 text-rose-600"
          )}>
            {isGood ? <TrendingDown className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3 rotate-180" />}
            {Math.round(percentage)}%
          </div>
        )}
      </div>

      <div className="flex items-end gap-6">
        <div className="space-y-1">
          <div className="text-[8px] font-bold text-slate-300 uppercase">Baseline</div>
          <div className="text-xl font-black text-slate-400">{Math.round(baseline)}</div>
        </div>
        
        <div className="h-8 w-px bg-slate-100 self-center" />

        <div className="space-y-1">
          <div className="text-[8px] font-bold text-indigo-400 uppercase tracking-wider">Humanized</div>
          <div className={clsx(
            "text-3xl font-black tracking-tighter",
            inverse 
              ? (humanized < 35 ? "text-emerald-500" : humanized > 60 ? "text-rose-500" : "text-amber-500")
              : "text-slate-900"
          )}>
            {Math.round(humanized)}
          </div>
        </div>
      </div>

      <div className="mt-5 space-y-2">
        <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden flex">
          <div 
            className="h-full bg-slate-300 opacity-40" 
            style={{ width: `${(baseline / (Math.max(baseline, humanized) || 1)) * 100}%` }} 
          />
        </div>
        <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden flex">
          <div 
            className={clsx("h-full", isGood ? "bg-emerald-500" : "bg-amber-500")} 
            style={{ width: `${(humanized / (Math.max(baseline, humanized) || 1)) * 100}%` }} 
          />
        </div>
      </div>
    </div>
  );
};

export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <div className="bg-slate-50 border-2 border-dashed border-slate-200 rounded-[2rem] p-10 flex flex-col items-center justify-center text-center space-y-4">
        <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-sm">
          <AlertCircle className="w-6 h-6 text-slate-300" />
        </div>
        <div className="space-y-1">
          <h4 className="text-sm font-black text-slate-400 uppercase tracking-widest">Metrics Unavailable</h4>
          <p className="text-xs text-slate-400 max-w-[200px]">Detailed statistical analysis was skipped for this segment.</p>
        </div>
      </div>
    );
  }

  const { baseline, humanized, latency } = metrics;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-500">
      <div className="flex items-center justify-between px-2">
        <div className="flex items-center gap-2.5">
          <BarChart3 className="w-4 h-4 text-indigo-600" />
          <h3 className="text-[10px] font-black text-slate-900 uppercase tracking-widest">Comparative Statistical Delta</h3>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-white rounded-lg border border-slate-200 text-[9px] font-bold text-slate-400">
          <Clock className="w-3 h-3" />
          Latency: <span className="text-slate-900">{latency.toFixed(2)}s</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard 
          label="AI Probability Score" 
          baseline={baseline.ai_score} 
          humanized={humanized.ai_score} 
          inverse={true} 
        />
        <MetricCard 
          label="Sentence Burstiness" 
          baseline={baseline.burstiness} 
          humanized={humanized.burstiness} 
          inverse={true} // In our detector, lower burstiness score means MORE variance (better)
        />
        <MetricCard 
          label="Lexical Perplexity" 
          baseline={baseline.perplexity} 
          humanized={humanized.perplexity} 
          inverse={true} // Lower score = higher unpredictability in our proxy
        />
      </div>

      <div className="bg-indigo-900 rounded-2xl p-4 flex items-center gap-4 text-white">
        <div className="w-10 h-10 bg-indigo-800 rounded-xl flex items-center justify-center shadow-inner">
          <Zap className="w-5 h-5 text-indigo-400" />
        </div>
        <div className="flex-1">
          <div className="text-[8px] font-black uppercase tracking-widest text-indigo-300">Optimization Summary</div>
          <div className="text-[11px] font-medium leading-relaxed">
            The pipeline achieved a <span className="text-emerald-400 font-bold">{Math.round(baseline.ai_score - humanized.ai_score)}% reduction</span> in machine-learned signatures by re-sculpting the rhythmic heartbeat of your manuscript.
          </div>
        </div>
        <Search className="w-5 h-5 text-indigo-700" />
      </div>
    </div>
  );
};
