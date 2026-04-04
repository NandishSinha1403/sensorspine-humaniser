"use client";

import { useState, useEffect, useMemo, useRef } from "react";
import { detectText, humanizeText, DetectionResponse, HumanizeResponse } from "./api";
import { MetricsDashboard } from "./components/MetricsDashboard";
import { 
  Loader2, AlertCircle, CheckCircle2, RefreshCw, Layers, Zap, PenTool, 
  ShieldCheck, Microscope, Gavel, Cpu, BookOpen, Briefcase, FileText, 
  History as HistoryIcon, Clock, Target, Type, Activity, Sparkles
} from "lucide-react";
import { clsx } from "clsx";


// Simple word-level diff algorithm
function computeWordDiff(original: string, modified: string): { word: string; type: "same" | "added" | "removed" }[] {
  const origWords = original.split(/\s+/);
  const modWords = modified.split(/\s+/);
  const result: { word: string; type: "same" | "added" | "removed" }[] = [];
  
  // LCS-based diff for proper word alignment
  const m = origWords.length;
  const n = modWords.length;
  
  // For performance, use a simplified approach for long texts
  if (m > 500 || n > 500) {
    // Set-based approach for long texts
    const origSet = new Set(origWords);
    for (const word of modWords) {
      result.push({ word, type: origSet.has(word) ? "same" : "added" });
    }
    return result;
  }
  
  // Build LCS table
  const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = origWords[i-1] === modWords[j-1] ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
    }
  }
  
  // Backtrack to build diff
  const diff: { word: string; type: "same" | "added" | "removed" }[] = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && origWords[i-1] === modWords[j-1]) {
      diff.unshift({ word: origWords[i-1], type: "same" });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
      diff.unshift({ word: modWords[j-1], type: "added" });
      j--;
    } else {
      diff.unshift({ word: origWords[i-1], type: "removed" });
      i--;
    }
  }
  
  return diff;
}

export default function Dashboard() {
  const [text, setText] = useState("");
  const [intensity, setIntensity] = useState(0.7);
  const [loading, setLoading] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [detection, setDetection] = useState<DetectionResponse | null>(null);
  const [humanizeResult, setHumanizeResult] = useState<HumanizeResponse | null>(null);
  const [showDiff, setShowDiff] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const resultRef = useRef<HTMLDivElement>(null);

  // Real-time Metrics — computed from live text input
  const stats = useMemo(() => {
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const sentences = text.trim() ? text.split(/[.!?]+/).filter(s => s.trim().length > 0).length : 0;
    const readingTime = Math.ceil(words / 200);
    const avgSentLen = sentences > 0 ? (words / sentences).toFixed(1) : 0;
    return { words, sentences, readingTime, avgSentLen };
  }, [text]);

  // Live Linguistic DNA metrics — computed from text
  const linguisticDNA = useMemo(() => {
    if (!text.trim()) return { variance: 0, complexity: 0, ttr: 0 };

    const words = text.trim().split(/\s+/);
    const sentenceTexts = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentLengths = sentenceTexts.map(s => s.trim().split(/\s+/).length);

    // Sentence length variance (std dev)
    const mean = sentLengths.reduce((a, b) => a + b, 0) / (sentLengths.length || 1);
    const variance = sentLengths.length > 1
      ? Math.sqrt(sentLengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / sentLengths.length)
      : 0;

    // Complexity = Flesch-Kincaid grade level approximation
    const syllables = words.reduce((sum, w) => {
      const cleaned = w.toLowerCase().replace(/[^a-z]/g, "");
      const count = cleaned.replace(/(?:[^laeio]es|ed|[^laeio]e)$/,"").replace(/^y/,"").match(/[aeiouy]{1,2}/g);
      return sum + (count ? count.length : 1);
    }, 0);
    const complexity = sentenceTexts.length > 0
      ? Math.max(1, Math.round(0.39 * (words.length / sentenceTexts.length) + 11.8 * (syllables / words.length) - 15.59))
      : 0;

    // Type-Token Ratio
    const uniqueWords = new Set(words.map(w => w.toLowerCase().replace(/[^a-z]/g, "")).filter(w => w.length > 0));
    const ttr = words.length > 0 ? (uniqueWords.size / words.length) : 0;

    return { variance: Math.round(variance * 10) / 10, complexity, ttr: Math.round(ttr * 100) / 100 };
  }, [text]);

  useEffect(() => {
    const saved = localStorage.getItem("humaniser_history");
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  useEffect(() => {
    if (humanizeResult && resultRef.current) {
      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    }
  }, [humanizeResult]);

  const saveToHistory = (item: any) => {
    const newHistory = [item, ...history].slice(0, 5);
    setHistory(newHistory);
    localStorage.setItem("humaniser_history", JSON.stringify(newHistory));
  };

  const handleHumanize = async () => {
    setLoading(true);
    setProcessingStep(1);
    setError(null);
    setHumanizeResult(null);
    
    const steps = [1, 2, 3, 4, 5, 6];
    for (const step of steps) {
      setProcessingStep(step);
      await new Promise(r => setTimeout(r, 300));
    }

    try {
      const res = await humanizeText(text, intensity);
      setHumanizeResult(res as any);
      setDetection(null);
      saveToHistory({ type: "humanize", preview: text.slice(0, 40), score: res.humanized_score, date: new Date().toLocaleTimeString() });
    } catch (e) {
      setError("Humanizer failed.");
    } finally {
      setLoading(false);
      setProcessingStep(0);
    }
  };

  const renderDiff = () => {
    if (!humanizeResult) return null;
    const diff = computeWordDiff(text, humanizeResult.humanized_text);
    return (
      <div className="serif-text text-lg leading-[1.8] text-slate-800 p-8 bg-white">
        {diff.map((item, i) => {
          if (item.type === "removed") {
            return (
              <span key={i} className="bg-rose-50 text-rose-500 line-through px-0.5 opacity-60">
                {item.word}{" "}
              </span>
            );
          }
          if (item.type === "added") {
            return (
              <span key={i} className="bg-emerald-50 text-emerald-700 border-b-2 border-emerald-200 px-0.5 font-semibold">
                {item.word}{" "}
              </span>
            );
          }
          return <span key={i}>{item.word}{" "}</span>;
        })}
      </div>
    );
  };

  const renderHeatmap = (sentences: any[]) => {
    if (!sentences) return null;
    return (
      <div className="serif-text text-lg leading-[1.8] text-slate-800 p-8 animate-in fade-in duration-500">
        {sentences.map((s, i) => {
          const opacity = s.score / 100;
          return (
            <span 
              key={i} 
              style={{ backgroundColor: `rgba(244, 63, 94, ${opacity * 0.3})` }}
              className={clsx("px-0.5 rounded transition-all cursor-help", s.score > 50 && "border-b border-rose-200")}
              title={`AI Probability: ${Math.round(s.score)}%`}
            >
              {s.text}{" "}
            </span>
          );
        })}
      </div>
    );
  };

  const getScoreColor = (score: number) => {
    if (score < 35) return "bg-emerald-500";
    if (score < 60) return "bg-amber-500";
    return "bg-rose-500";
  };

  return (
    <div className="bg-[#F4F7F9] min-h-screen font-sans selection:bg-indigo-100 selection:text-indigo-900">
      <div className="max-w-[1500px] mx-auto p-4 lg:p-6 pb-20">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 px-2">
          <div className="space-y-0.5">
            <h1 className="text-2xl font-black text-slate-900 tracking-tight flex items-center gap-2.5">
              <Activity className="text-indigo-600 w-6 h-6" /> Manuscript Intelligence
            </h1>
            <p className="text-slate-400 text-[10px] font-bold uppercase tracking-widest">Professional Linguistic Suite</p>
          </div>
          <div className="bg-white/80 backdrop-blur-sm px-4 py-2 rounded-xl border border-slate-200 shadow-sm flex items-center gap-6 mt-4 md:mt-0">
             <div className="flex flex-col">
                <span className="text-[9px] font-black text-slate-400 uppercase tracking-tighter">Words</span>
                <span className="text-sm font-black text-slate-900">{stats.words}</span>
             </div>
             <div className="w-px h-6 bg-slate-100" />
             <div className="flex flex-col">
                <span className="text-[9px] font-black text-slate-400 uppercase tracking-tighter">Read Time</span>
                <span className="text-sm font-black text-slate-900">{stats.readingTime}m</span>
             </div>
             <div className="w-px h-6 bg-slate-100" />
             <div className="flex flex-col">
                <span className="text-[9px] font-black text-slate-400 uppercase tracking-tighter">Complexity</span>
                <span className="text-sm font-black text-slate-900">{stats.avgSentLen} <span className="text-[10px] text-slate-400 font-normal italic">w/s</span></span>
             </div>
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-6">
          
          {/* Main Workspace */}
          <div className="flex-1 space-y-6">
            <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/40 border border-slate-200 overflow-hidden flex flex-col h-[500px] md:h-[600px] lg:h-[750px]">
              

              {/* Editor */}
              <div className="relative flex-1 group bg-[#FDFDFD]">
                <textarea
                  className="w-full h-full p-6 md:p-10 serif-text text-base md:text-lg leading-[1.8] text-slate-800 focus:outline-none placeholder:text-slate-200 resize-none bg-transparent min-h-[150px] md:min-h-[180px] lg:min-h-[200px]"
                  placeholder="Paste manuscript draft for analysis and humanization..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
              </div>

              {/* Control Bar */}
              <div className="bg-slate-50/80 border-t border-slate-200 p-4 md:p-6 flex flex-col lg:flex-row items-center justify-between gap-6 md:gap-8">
                <div className="space-y-3 w-full lg:w-80">
                   <div className="flex justify-between items-center px-1">
                      <span className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-400 flex items-center gap-2">
                        <Sparkles className="w-3 h-3 text-indigo-500" /> Intensity
                      </span>
                      <span className="text-indigo-600 font-black text-xs">{Math.round(intensity * 100)}%</span>
                   </div>
                   <div className="relative h-1.5 bg-slate-200 rounded-full overflow-hidden">
                      <div className="absolute inset-y-0 left-0 bg-indigo-600" style={{ width: `${intensity * 100}%` }} />
                      <input
                        type="range" min="0.1" max="1.0" step="0.1"
                        className="absolute inset-0 w-full opacity-0 cursor-pointer"
                        value={intensity}
                        onChange={(e) => setIntensity(parseFloat(e.target.value))}
                      />
                   </div>
                </div>

                <div className="flex flex-col md:flex-row gap-3 w-full lg:w-auto items-center">
                  {error && (
                    <div className="flex items-center gap-2 text-rose-500 text-[10px] font-bold uppercase mr-4 bg-rose-50 px-3 py-2 rounded-lg border border-rose-100">
                      <AlertCircle className="w-3.5 h-3.5" /> {error}
                    </div>
                  )}
                  <button
                    onClick={async () => {
                       setLoading(true);
                       setError(null);
                       await detectText(text).then(res => { setDetection(res as any); setHumanizeResult(null); setLoading(false); }).catch(e => { setError("Analysis failed"); setLoading(false); });
                    }}
                    disabled={loading || !text}
                    className="w-full md:px-10 bg-white hover:bg-slate-50 text-slate-900 border border-slate-200 font-black text-[10px] uppercase tracking-widest py-4 rounded-xl transition-all disabled:opacity-40"
                  >
                    Analyze
                  </button>
                  <button
                    onClick={handleHumanize}
                    disabled={loading || !text}
                    className="w-full md:px-14 bg-indigo-600 hover:bg-indigo-700 text-white font-black text-[10px] uppercase tracking-[0.2em] py-4 rounded-xl transition-all shadow-lg shadow-indigo-100 disabled:opacity-40 flex items-center justify-center gap-3"
                  >
                    {loading ? <Loader2 className="animate-spin w-4 h-4" /> : <><RefreshCw className="w-4 h-4" /> Humanize &amp; Refine</>}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right Sidebar */}
          <div className="w-full md:w-[320px] lg:w-[380px] space-y-5">
            
            {/* Terminal */}
            <div className="bg-slate-900 rounded-[2rem] shadow-xl p-6 text-white relative overflow-hidden">
               <div className="relative z-10 space-y-5">
                  <div className="flex items-center justify-between">
                     <span className="text-[9px] font-black uppercase tracking-[0.2em] text-indigo-400">Analysis Engine</span>
                     <div className="flex gap-1">
                        <div className="w-1 h-1 rounded-full bg-emerald-500 animate-pulse" />
                        <div className="w-1 h-1 rounded-full bg-slate-700" />
                     </div>
                  </div>

                  {loading ? (
                    <div className="space-y-3 py-4 font-mono text-[9px] text-slate-500 uppercase tracking-widest">
                       {[ "Citation Guard", "Confidence Gradient", "Structural Burstiness", "Lexical Mapping", "Voice Conversion", "Self-Audit Pass" ].map((step, idx) => (
                         <div key={idx} className={clsx("flex items-center gap-2 transition-opacity", processingStep > idx ? "opacity-100 text-emerald-400" : "opacity-20")}>
                            {processingStep > idx ? <CheckCircle2 className="w-3 h-3"/> : <Loader2 className="w-3 h-3 animate-spin"/>}
                            {step}...
                         </div>
                       ))}
                    </div>
                  ) : (
                    <div className="space-y-5">
                       <div className="flex items-end gap-3">
                          <span className="text-4xl md:text-5xl font-black tracking-tighter">
                             {Math.round(humanizeResult ? humanizeResult.humanized_score : (detection?.score || 0))}
                          </span>
                          <div className="pb-1.5">
                             <div className="text-[10px] font-bold text-indigo-400 uppercase">AI Prob</div>
                             <div className="text-[8px] text-slate-500 font-bold uppercase tracking-widest leading-none">Confidence {Math.round(100 - (humanizeResult?.humanized_score || 0))}%</div>
                          </div>
                       </div>
                       
                       {/* Score Comparison Bars */}
                       <div className="flex flex-col md:grid md:grid-cols-2 gap-3">
                          {humanizeResult && (
                            <div className="space-y-1">
                               <div className="flex justify-between text-[8px] font-bold uppercase text-slate-500">
                                  <span>Original Score</span>
                                  <span>{Math.round(humanizeResult.original_score)}%</span>
                                </div>
                               <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                                  <div className="h-full bg-slate-600 opacity-50" style={{ width: `${humanizeResult.original_score}%` }} />
                               </div>
                            </div>
                          )}
                          <div className="space-y-1">
                             <div className="flex justify-between text-[8px] font-bold uppercase text-indigo-400">
                                <span>{humanizeResult ? "Humanized Score" : "Current AI Score"}</span>
                                <span>{Math.round(humanizeResult ? humanizeResult.humanized_score : (detection?.score || 0))}%</span>
                              </div>
                             <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                                <div 
                                  className={clsx("h-full transition-all duration-1000", getScoreColor(humanizeResult ? humanizeResult.humanized_score : (detection?.score || 0)))}
                                  style={{ width: `${humanizeResult ? humanizeResult.humanized_score : (detection?.score || 0)}%` }}
                                />
                             </div>
                          </div>
                       </div>

                       <div className="grid grid-cols-2 gap-3">
                          <button 
                            onClick={() => { setShowHeatmap(!showHeatmap); setShowDiff(false); }}
                            className={clsx("py-2.5 rounded-xl text-[9px] font-black uppercase tracking-widest transition-all border", 
                              showHeatmap ? "bg-indigo-600 border-indigo-600 text-white" : "bg-transparent text-slate-500 border-slate-800 hover:border-slate-600")}
                          >
                            AI Heatmap
                          </button>
                          <button 
                            disabled={!humanizeResult}
                            onClick={() => { setShowDiff(!showDiff); setShowHeatmap(false); }}
                            className={clsx("py-2.5 rounded-xl text-[9px] font-black uppercase tracking-widest transition-all border", 
                              showDiff ? "bg-emerald-600 border-emerald-600 text-white" : "bg-transparent text-slate-500 border-slate-800 hover:border-slate-600 disabled:opacity-10")}
                          >
                            Show Diff
                          </button>
                       </div>
                    </div>
                  )}
               </div>
            </div>

            {/* Detailed Metrics Delta */}
            {humanizeResult && (
              <MetricsDashboard metrics={humanizeResult.metrics} />
            )}

            {/* Changes Made Stats */}
            {humanizeResult && (
              <div className="space-y-4">
                <div className="bg-slate-50 rounded-2xl border border-slate-200 p-4 space-y-2">
                  <div className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-2">Refinement Stats</div>
                  <div className="text-[10px] font-bold text-slate-700 flex flex-wrap md:flex-nowrap gap-x-3 gap-y-1 overflow-x-auto no-scrollbar">
                    <span className="whitespace-nowrap">{humanizeResult.changes_made.phrases_replaced || 0} phrases replaced</span>
                    <span className="hidden md:inline">•</span>
                    <span className="whitespace-nowrap">{humanizeResult.changes_made.lexical_substitutions || 0} words substituted</span>
                    <span className="hidden md:inline">•</span>
                    <span className="whitespace-nowrap">{humanizeResult.changes_made.sentences_restructured || 0} restructured</span>
                    {(humanizeResult.changes_made.voice_conversions || 0) > 0 && (
                      <>
                        <span className="hidden md:inline">•</span>
                        <span className="whitespace-nowrap">{humanizeResult.changes_made.voice_conversions} voice flips</span>
                      </>
                    )}
                    {(humanizeResult.changes_made.clause_reorders || 0) > 0 && (
                      <>
                        <span className="hidden md:inline">•</span>
                        <span className="whitespace-nowrap">{humanizeResult.changes_made.clause_reorders} clause reorders</span>
                      </>
                    )}
                    {(humanizeResult.changes_made.audit_iterations || 0) > 0 && (
                      <>
                        <span className="hidden md:inline">•</span>
                        <span className="whitespace-nowrap">{humanizeResult.changes_made.audit_iterations} audit loops</span>
                      </>
                    )}
                  </div>
                </div>

                {humanizeResult.voice_profile && (
                  <div className="bg-white rounded-2xl border border-slate-200 p-4 space-y-3 shadow-sm border-l-4 border-l-indigo-500">
                    <div className="flex items-center gap-2">
                       <HistoryIcon className="w-3 h-3 text-indigo-500" />
                       <div className="text-[8px] font-black text-slate-400 uppercase tracking-widest">Detected Writing Voice</div>
                    </div>
                    <div className="grid grid-cols-2 gap-y-3">
                       <div>
                          <div className="text-[7px] font-black text-slate-300 uppercase">Style</div>
                          <div className="text-[9px] md:text-[10px] font-black text-slate-900 capitalize">
                             {humanizeResult.voice_profile.formality_score > 0.1 ? 'Formal' : humanizeResult.voice_profile.formality_score > 0.05 ? 'Semi-Formal' : 'Informal'}
                          </div>
                       </div>
                       <div>
                          <div className="text-[7px] font-black text-slate-300 uppercase">Person</div>
                          <div className="text-[9px] md:text-[10px] font-black text-slate-900 capitalize">{humanizeResult.voice_profile.person.replace('_', ' ')}</div>
                       </div>
                       <div className="col-span-2">
                          <div className="text-[7px] font-black text-slate-300 uppercase">Sentence Architecture</div>
                          <div className="text-[9px] md:text-[10px] font-black text-slate-900">Avg {Math.round(humanizeResult.voice_profile.preferred_sentence_length)} words / sentence</div>
                       </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Linguistic DNA — LIVE from input text */}
            <div className="bg-white rounded-[2rem] shadow-lg border border-slate-200 p-6 space-y-6">
               <div className="flex items-center gap-2.5">
                  <Target className="w-4 h-4 text-indigo-600" />
                  <h3 className="text-[10px] font-black text-slate-900 uppercase tracking-widest">Linguistic DNA</h3>
               </div>
               
               <div className="space-y-5">
                  <div className="space-y-2">
                     <div className="flex justify-between text-[8px] font-black uppercase text-slate-400 tracking-wider">
                        <span>Sentence Length Variance</span>
                        <span className="text-slate-900">{linguisticDNA.variance}</span>
                     </div>
                     <div className="flex gap-1 h-6 items-end px-1">
                        {text.trim() ? (
                          text.split(/[.!?]+/).filter(s => s.trim().length > 0).slice(0, 10).map((s, i) => {
                            const wc = s.trim().split(/\s+/).length;
                            const maxWc = 40;
                            const h = Math.min(100, (wc / maxWc) * 100);
                            return <div key={i} className="flex-1 bg-slate-100 rounded-t-[1px] transition-all hover:bg-indigo-400" style={{ height: `${h}%` }} />;
                          })
                        ) : (
                          [30, 60, 40, 80, 55, 25, 70, 45, 55, 35].map((h, i) => (
                            <div key={i} className="flex-1 bg-slate-100 rounded-t-[1px] transition-all hover:bg-indigo-400" style={{ height: `${h}%` }} />
                          ))
                        )}
                     </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                     <div className="p-3 bg-slate-50 rounded-2xl border border-slate-100">
                        <div className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-0.5">Complexity</div>
                        <div className="text-lg font-black text-slate-900">Lvl {linguisticDNA.complexity}</div>
                     </div>
                     <div className="p-3 bg-slate-50 rounded-2xl border border-slate-100">
                        <div className="text-[8px] font-black text-slate-400 uppercase tracking-widest mb-0.5">TTR Ratio</div>
                        <div className="text-lg font-black text-slate-900">{linguisticDNA.ttr}</div>
                     </div>
                  </div>
               </div>
            </div>

            {/* History Small */}
            <div className="px-2 space-y-3">
               <h4 className="text-[9px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                  <Clock className="w-3 h-3"/> Recent Sessions
               </h4>
               <div className="space-y-2">
                  {history.slice(0, 3).map((h, i) => (
                     <div key={i} className="flex items-center justify-between p-3 bg-white rounded-xl border border-slate-200 text-[10px] font-bold text-slate-600 shadow-sm">
                        <span className="truncate w-32 font-medium italic opacity-60">"{h.preview}..."</span>
                        <span className="text-indigo-600">{h.score}%</span>
                     </div>
                  ))}
               </div>
            </div>

          </div>
        </div>

        {/* Dedicated Humanized Output Section - Full Width at Bottom */}
        <div ref={resultRef}>
          {humanizeResult && (
            <div className="mt-8 bg-white rounded-3xl shadow-xl shadow-slate-200/40 border border-slate-200 overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-700">
              <div className="bg-slate-900 px-6 md:px-10 py-5 flex flex-col sm:flex-row justify-between items-center gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center text-white">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                  <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
                    <h2 className="text-sm font-black text-white uppercase tracking-widest">
                      Humanized Output
                    </h2>
                    <div className="flex items-center gap-2">
                       <span className="bg-emerald-500/20 text-emerald-400 text-[9px] font-black px-2 py-0.5 rounded border border-emerald-500/30 uppercase">
                         AI Score: {Math.round(humanizeResult.humanized_score)}%
                       </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4 w-full sm:w-auto justify-between sm:justify-end">
                  <div className="flex items-center gap-3 text-[9px] font-bold text-slate-400 uppercase tracking-widest bg-slate-800/50 px-3 py-1.5 rounded-lg border border-slate-700">
                    <span>Original: <span className="text-slate-300">{Math.round(humanizeResult.original_score)}</span></span>
                    <span className="text-indigo-500">→</span>
                    <span>Human: <span className="text-emerald-400">{Math.round(humanizeResult.humanized_score)}</span></span>
                  </div>
                  <button 
                    onClick={() => {
                      navigator.clipboard.writeText(humanizeResult.humanized_text);
                    }}
                    className="bg-white text-slate-900 px-4 md:px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-emerald-50 transition-all flex items-center gap-2 shadow-lg"
                  >
                    <FileText className="w-3 h-3 text-emerald-600" />
                    Copy
                  </button>
                </div>
              </div>
              
              <div className="bg-white min-h-[400px] relative">
                <div className="serif-text text-lg leading-[1.8] text-slate-800 p-8 md:p-12 whitespace-pre-wrap selection:bg-emerald-100 max-h-[800px] overflow-y-auto">
                  {humanizeResult.humanized_text}
                </div>
              </div>

              <div className="bg-slate-50/50 px-6 md:px-10 py-4 border-t border-slate-100 flex flex-wrap gap-4 items-center justify-between">
                 <div className="flex gap-4">
                    <button 
                      onClick={() => { setShowHeatmap(!showHeatmap); setShowDiff(false); }}
                      className={clsx("text-[9px] font-black uppercase tracking-widest px-3.5 py-2 rounded-lg border transition-all", 
                        showHeatmap ? "bg-rose-50 border-rose-200 text-rose-600 shadow-sm" : "bg-white border-slate-200 text-slate-500 hover:border-indigo-200")}
                    >
                      AI Heatmap
                    </button>
                    <button 
                      onClick={() => { setShowDiff(!showDiff); setShowHeatmap(false); }}
                      className={clsx("text-[9px] font-black uppercase tracking-widest px-3.5 py-2 rounded-lg border transition-all", 
                        showDiff ? "bg-emerald-50 border-emerald-200 text-emerald-600 shadow-sm" : "bg-white border-slate-200 text-slate-500 hover:border-indigo-200")}
                    >
                      Show Diff
                    </button>
                 </div>
                 <div className="text-[10px] font-bold text-slate-400 italic">
                    Verified Academic Manuscript Refinement
                 </div>
              </div>

              {(showHeatmap || showDiff) && (
                <div className="bg-slate-50 border-t border-slate-200 animate-in slide-in-from-top-2 duration-300">
                  {showHeatmap ? renderHeatmap(humanizeResult.sentences as any) : renderDiff()}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
