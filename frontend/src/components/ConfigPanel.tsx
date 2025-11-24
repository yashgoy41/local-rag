import React from 'react';
import { Settings } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface Model {
    model: string;
}

interface Config {
    chunkingModel: string;
    embeddingModel: string;
    rerankerModel: string;
    generationModel: string;
}

interface ConfigPanelProps {
    models: Model[];
    config: Config;
    setConfig: React.Dispatch<React.SetStateAction<Config>>;
}

export default function ConfigPanel({ models, config, setConfig }: ConfigPanelProps) {
    const handleChange = (key: keyof Config, value: string) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    };

    return (
        <Card>
            <CardHeader>
                <div className="flex items-center gap-2">
                    <Settings size={20} className="text-black" />
                    <CardTitle>Configuration</CardTitle>
                </div>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Chunking */}
                <div className="space-y-2">
                    <Label htmlFor="chunking-model">Chunking Model</Label>
                    <Select value={config.chunkingModel} onValueChange={(value) => handleChange('chunkingModel', value)}>
                        <SelectTrigger id="chunking-model">
                            <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                            {models.map(m => (
                                <SelectItem key={m.model} value={m.model}>{m.model}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                {/* Embedding */}
                <div className="space-y-2">
                    <Label htmlFor="embedding-model">Embedding Model</Label>
                    <Select value={config.embeddingModel} onValueChange={(value) => handleChange('embeddingModel', value)}>
                        <SelectTrigger id="embedding-model">
                            <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                            {models.map(m => (
                                <SelectItem key={m.model} value={m.model}>{m.model}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                {/* Reranking */}
                <div className="space-y-2">
                    <Label htmlFor="reranker-model">Reranker Model</Label>
                    <Select value={config.rerankerModel} onValueChange={(value) => handleChange('rerankerModel', value)}>
                        <SelectTrigger id="reranker-model">
                            <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="BAAI/bge-reranker-v2-m3">BAAI/bge-reranker-v2-m3</SelectItem>
                        </SelectContent>
                    </Select>
                </div>

                {/* Generation */}
                <div className="space-y-2">
                    <Label htmlFor="generation-model">Generation Model</Label>
                    <Select value={config.generationModel} onValueChange={(value) => handleChange('generationModel', value)}>
                        <SelectTrigger id="generation-model">
                            <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                            {models.map(m => (
                                <SelectItem key={m.model} value={m.model}>{m.model}</SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            </CardContent>
        </Card>
    );
}
