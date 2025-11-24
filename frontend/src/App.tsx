import { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Play, Database, Search, MessageSquare, BarChart3 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';

import ConfigPanel from './components/ConfigPanel';
import UploadZone from './components/UploadZone';
import BenchmarkView from './components/BenchmarkView';

const API_URL = 'http://localhost:8000';

interface Config {
    chunkingModel: string;
    embeddingModel: string;
    rerankerModel: string;
    generationModel: string;
}

interface Metrics {
    ingestion?: { total_time: number };
    embedding?: { total_time: number };
    retrieval?: { total_time: number };
    generation?: { generation_time: number; token_count?: number; tokens_per_sec?: number };
}

interface Result {
    text: string;
    source: string;
    page: number;
    score: number;
}

function App() {
    const [models, setModels] = useState<any[]>([]);
    const [config, setConfig] = useState<Config>({
        chunkingModel: '',
        embeddingModel: '',
        rerankerModel: 'BAAI/bge-reranker-v2-m3',
        generationModel: ''
    });
    const [file, setFile] = useState<File | null>(null);
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<Result[] | null>(null);
    const [answer, setAnswer] = useState('');
    const [metrics, setMetrics] = useState<Metrics>({});
    const [loading, setLoading] = useState(false);
    const [step, setStep] = useState<'upload' | 'process' | 'retrieve' | 'generate'>('upload');
    const [progress, setProgress] = useState({ stage: '', current: 0, total: 0 });
    const [streamingTokensPerSec, setStreamingTokensPerSec] = useState(0);
    const [isStreaming, setIsStreaming] = useState(false);

    useEffect(() => {
        const init = async () => {
            try { await axios.post(`${API_URL}/reset`); } catch (e) { console.error(e); }
            fetchModels();
        };
        init();
    }, []);

    const fetchModels = async () => {
        try {
            const res = await axios.get(`${API_URL}/models`);
            setModels(res.data.models);
            const embeddingModel = res.data.embedding_models?.[0]?.model || 'qwen3-embedding:4b';
            const generationModel = res.data.generation_models?.[0]?.model || res.data.models[0]?.model;
            setConfig(prev => ({
                ...prev,
                chunkingModel: generationModel,
                embeddingModel,
                generationModel
            }));
        } catch (e) { console.error(e); }
    };

    const handleUpload = async (uploadedFile: File) => {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        try {
            setLoading(true);
            await axios.post(`${API_URL}/upload`, formData);
            setFile(uploadedFile);
            setStep('process');
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const handleProcess = async () => {
        if (!file) return;
        try {
            setLoading(true);
            setProgress({ stage: 'Processing document...', current: 0, total: 0 });

            const res = await axios.post(`${API_URL}/process`, {
                filename: file.name,
                chunking_model: config.chunkingModel
            });
            setMetrics(prev => ({ ...prev, ingestion: res.data.metrics }));
            setProgress({ stage: 'Generating embeddings...', current: 0, total: res.data.chunks_count });

            const embedRes = await axios.post(`${API_URL}/embed`, { embedding_model: config.embeddingModel });
            setMetrics(prev => ({ ...prev, embedding: embedRes.data.metrics }));
            setProgress({ stage: '', current: 0, total: 0 });
            setStep('retrieve');
        } catch (e) { console.error(e); setProgress({ stage: '', current: 0, total: 0 }); }
        finally { setLoading(false); }
    };

    const handleRetrieve = async () => {
        if (!query) return;
        try {
            setLoading(true);
            const res = await axios.post(`${API_URL}/retrieve`, {
                query,
                embedding_model: config.embeddingModel,
                reranker_model: config.rerankerModel
            });
            setResults(res.data.results);
            setMetrics(prev => ({ ...prev, retrieval: res.data.metrics }));
            setStep('generate');
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const handleGenerate = async () => {
        if (!results) return;
        try {
            setLoading(true);
            setIsStreaming(true);
            setAnswer('');
            setStreamingTokensPerSec(0);
            
            const context = results.map(r => r.text).join("\n\n");
            const response = await fetch(`${API_URL}/generate/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, context, model: config.generationModel })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const reader = response.body?.getReader();
            const decoder = new TextDecoder();
            if (!reader) throw new Error('No reader');

            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.content) setAnswer(prev => prev + data.content);
                        if (data.tokens_per_sec !== undefined) setStreamingTokensPerSec(data.tokens_per_sec);
                        if (data.done && data.metrics) setMetrics(prev => ({ ...prev, generation: data.metrics }));
                    } catch { /* skip */ }
                }
            }
        } catch (err) {
            console.error("Streaming failed, trying fallback:", err);
            try {
                const context = results.map(r => r.text).join("\n\n");
                const res = await axios.post(`${API_URL}/generate`, { query, context, model: config.generationModel });
                setAnswer(res.data.answer);
                setMetrics(prev => ({ ...prev, generation: res.data.metrics }));
            } catch (e) { console.error(e); }
        } finally {
            setLoading(false);
            setIsStreaming(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
            <header className="mb-8 flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-black text-white shadow-lg">
                    <BarChart3 size={20} />
                </div>
                <div>
                    <h1 className="text-2xl font-bold tracking-tight">RAG Benchmarking Dashboard</h1>
                    <p className="text-slate-500">Test and optimize your local RAG pipeline</p>
                </div>
            </header>

            <main className="grid grid-cols-12 gap-8">
                <div className="col-span-3 space-y-6">
                    <ConfigPanel models={models} config={config} setConfig={setConfig} />
                </div>

                <div className="col-span-9 space-y-6">
                    {/* Upload & Ingest */}
                    <Card>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <Upload size={20} />
                                    <CardTitle>Document Ingestion</CardTitle>
                                </div>
                                {metrics.ingestion && (
                                    <Badge variant="outline" className="text-green-600">
                                        {metrics.ingestion.total_time.toFixed(2)}s
                                    </Badge>
                                )}
                            </div>
                        </CardHeader>
                        <CardContent>
                            {!file ? (
                                <UploadZone onUpload={handleUpload} loading={loading} />
                            ) : (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between rounded-lg border bg-slate-50 p-4">
                                        <div className="flex items-center gap-3">
                                            <div className="flex h-10 w-10 items-center justify-center rounded bg-white shadow-sm">📄</div>
                                            <div>
                                                <p className="font-medium">{file.name}</p>
                                                <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</p>
                                            </div>
                                        </div>
                                        {step === 'process' && (
                                            <Button onClick={handleProcess} disabled={loading}>
                                                {loading ? 'Processing...' : 'Start Processing'}
                                                {!loading && <Play size={16} className="ml-2" />}
                                            </Button>
                                        )}
                                        {step !== 'process' && step !== 'upload' && (
                                            <Badge className="bg-green-600">
                                                <Database size={16} className="mr-1" />
                                                Ingested
                                            </Badge>
                                        )}
                                    </div>
                                    {loading && progress.stage && (
                                        <div className="rounded-lg border bg-slate-50 p-4">
                                            <p className="text-sm font-medium mb-2">{progress.stage}</p>
                                            <Progress value={progress.total > 0 ? (progress.current / progress.total) * 100 : 0} className="h-2" />
                                        </div>
                                    )}
                                </div>
                            )}
                        </CardContent>
                    </Card>

                    {/* Retrieval */}
                    {metrics.embedding && (
                        <Card>
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <Search size={20} />
                                        <CardTitle>Retrieval</CardTitle>
                                    </div>
                                    {metrics.retrieval && (
                                        <Badge variant="outline" className="text-green-600">
                                            {metrics.retrieval.total_time.toFixed(2)}s
                                        </Badge>
                                    )}
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="flex gap-2">
                                    <Input
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && !loading && query && handleRetrieve()}
                                        placeholder="Enter your query..."
                                        className="flex-1"
                                    />
                                    <Button onClick={handleRetrieve} disabled={loading || !query}>
                                        {loading ? 'Searching...' : 'Search'}
                                    </Button>
                                </div>
                                {results && results.length > 0 && (
                                    <div className="space-y-3">
                                        <h3 className="text-sm font-medium text-slate-500">Top Chunks</h3>
                                        {results.map((r, i) => (
                                            <div key={i} className="rounded-lg border bg-slate-50 p-4">
                                                <div className="mb-2 flex justify-between text-xs text-slate-500">
                                                    <span>Score: {r.score.toFixed(4)}</span>
                                                    <span>Page {r.page}</span>
                                                </div>
                                                <p className="text-sm text-slate-700">{r.text}</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}

                    {/* Generation */}
                    {results && results.length > 0 && (
                        <Card>
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <MessageSquare size={20} />
                                        <CardTitle>Generation</CardTitle>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        {isStreaming && (
                                            <Badge variant="outline" className="font-mono text-blue-600">
                                                {streamingTokensPerSec.toFixed(1)} tok/s
                                            </Badge>
                                        )}
                                        {metrics.generation ? (
                                            <Badge variant="outline" className="text-green-600">
                                                {metrics.generation.tokens_per_sec?.toFixed(1)} tok/s · {metrics.generation.generation_time.toFixed(2)}s
                                            </Badge>
                                        ) : (
                                            <Button onClick={handleGenerate} disabled={loading} size="sm">
                                                {loading ? 'Generating...' : 'Generate'}
                                                {!loading && <Play size={16} className="ml-2" />}
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            </CardHeader>
                            {(answer || isStreaming) && (
                                <CardContent>
                                    <div className="rounded-lg bg-slate-50 p-6">
                                        <p className="whitespace-pre-wrap">
                                            {answer}
                                            {isStreaming && <span className="inline-block w-2 h-4 ml-1 bg-slate-400 animate-pulse" />}
                                        </p>
                                    </div>
                                </CardContent>
                            )}
                        </Card>
                    )}

                    <BenchmarkView metrics={metrics} />
                </div>
            </main>
        </div>
    );
}

export default App;
