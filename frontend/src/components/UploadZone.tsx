import React from 'react';
import { Upload } from 'lucide-react';
import { cn } from '@/lib/utils';

interface UploadZoneProps {
    onUpload: (file: File) => void;
    loading: boolean;
}

export default function UploadZone({ onUpload, loading }: UploadZoneProps) {
    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        if (loading) return;

        const file = e.dataTransfer.files[0];
        if (file) onUpload(file);
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (loading) return;
        const file = e.target.files?.[0];
        if (file) onUpload(file);
    };

    return (
        <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className={cn(
                "relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 py-12 transition-colors hover:border-indigo-500 hover:bg-indigo-50/50",
                loading && "cursor-not-allowed opacity-50"
            )}
        >
            <input
                type="file"
                onChange={handleChange}
                className="absolute inset-0 cursor-pointer opacity-0"
                disabled={loading}
            />
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-indigo-100 text-indigo-600">
                <Upload size={24} />
            </div>
            <p className="mt-4 text-sm font-medium text-slate-900">
                Click to upload or drag and drop
            </p>
            <p className="mt-1 text-xs text-slate-500">PDF or TXT files</p>
        </div>
    );
}
