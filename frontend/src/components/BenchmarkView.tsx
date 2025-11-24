import React from 'react';
import { BarChart3, Zap, Clock } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface Metrics {
    ingestion?: {
        total_time: number;
    };
    embedding?: {
        total_time: number;
    };
    retrieval?: {
        total_time: number;
    };
    generation?: {
        generation_time: number;
        token_count?: number;
        tokens_per_sec?: number;
    };
}

interface BenchmarkViewProps {
    metrics: Metrics;
}

export default function BenchmarkView({ metrics }: BenchmarkViewProps) {
    if (Object.keys(metrics).length === 0) return null;

    // Calculate total pipeline time
    const totalTime = 
        (metrics.ingestion?.total_time || 0) + 
        (metrics.embedding?.total_time || 0) + 
        (metrics.retrieval?.total_time || 0) + 
        (metrics.generation?.generation_time || 0);

    return (
        <Card>
            <CardHeader>
                <div className="flex items-center gap-2">
                    <BarChart3 size={20} className="text-black" />
                    <CardTitle>Performance Summary</CardTitle>
                </div>
            </CardHeader>
            <CardContent>
                {/* Primary Metrics - What Actually Matters */}
                <div className="grid grid-cols-2 gap-6 mb-6">
                    {/* Total Pipeline Time */}
                    <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-slate-50 to-white p-6">
                        <div className="flex items-center gap-2 text-slate-500 mb-2">
                            <Clock size={16} />
                            <span className="text-sm font-medium">Total Pipeline Time</span>
                        </div>
                        <p className="text-4xl font-bold text-slate-900 tracking-tight">
                            {totalTime.toFixed(2)}<span className="text-xl font-normal text-slate-400">s</span>
                        </p>
                    </div>

                    {/* Generation Throughput */}
                    {metrics.generation && (
                        <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-emerald-50 to-white p-6">
                            <div className="flex items-center gap-2 text-emerald-600 mb-2">
                                <Zap size={16} />
                                <span className="text-sm font-medium">Generation Speed</span>
                            </div>
                            <p className="text-4xl font-bold text-slate-900 tracking-tight">
                                {metrics.generation.tokens_per_sec?.toFixed(1) || '—'}
                                <span className="text-xl font-normal text-slate-400"> tok/s</span>
                            </p>
                            {metrics.generation.token_count && (
                                <p className="text-xs text-slate-400 mt-1">
                                    {metrics.generation.token_count} tokens generated
                                </p>
                            )}
                        </div>
                    )}
                </div>

                {/* Stage Breakdown - Compact */}
                <div className="border-t border-slate-100 pt-4">
                    <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3">Stage Breakdown</p>
                    <div className="flex gap-2 flex-wrap">
                        {metrics.ingestion && (
                            <StageChip label="Ingest" time={metrics.ingestion.total_time} />
                        )}
                        {metrics.embedding && (
                            <StageChip label="Embed" time={metrics.embedding.total_time} />
                        )}
                        {metrics.retrieval && (
                            <StageChip label="Retrieve" time={metrics.retrieval.total_time} />
                        )}
                        {metrics.generation && (
                            <StageChip label="Generate" time={metrics.generation.generation_time} />
                        )}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

function StageChip({ label, time }: { label: string; time: number }) {
    return (
        <div className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5">
            <span className="text-xs font-medium text-slate-600">{label}</span>
            <span className="text-xs font-bold text-slate-900">{time.toFixed(2)}s</span>
        </div>
    );
}
