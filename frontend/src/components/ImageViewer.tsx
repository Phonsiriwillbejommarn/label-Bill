import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Image as ImageIcon } from 'lucide-react';

interface ImageViewerProps {
    imageSrc: string | null;
    imageName: string | null;
    currentIndex: number;
    totalImages: number;
    onPrev: () => void;
    onNext: () => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
    imageSrc,
    imageName,
    currentIndex,
    totalImages,
    onPrev,
    onNext
}) => {
    const [zoom, setZoom] = useState(1);

    // Reset zoom when image changes
    useEffect(() => {
        setZoom(1);
    }, [imageSrc]);

    const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 3));
    const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.5));

    return (
        <div className="glass-panel flex-1 flex flex-col relative overflow-hidden group h-full">
            {/* Counter Badge */}
            <div className="absolute top-4 left-4 z-10 bg-black/50 backdrop-blur px-3 py-1 rounded-full text-xs font-medium text-white border border-white/10">
                <span>{totalImages > 0 ? currentIndex + 1 : 0} / {totalImages}</span>
            </div>

            {/* Image Container */}
            <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] bg-slate-900/50">
                {imageSrc ? (
                    <motion.img
                        src={imageSrc}
                        alt={imageName || 'Receipt'}
                        className="max-h-full max-w-full object-contain transition-transform duration-200"
                        style={{ transform: `scale(${zoom})` }}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: zoom }}
                        transition={{ duration: 0.3 }}
                        drag
                        dragConstraints={{ left: -200, right: 200, top: -200, bottom: 200 }}
                    />
                ) : (
                    <div className="text-center text-slate-500 flex flex-col items-center">
                        <ImageIcon className="w-16 h-16 mb-4 opacity-50" />
                        <p>No images loaded</p>
                    </div>
                )}
            </div>

            {/* Zoom Controls */}
            <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <button onClick={handleZoomOut} className="glass-btn p-2 rounded-full"><ZoomOut size={20} /></button>
                <button onClick={handleZoomIn} className="glass-btn p-2 rounded-full"><ZoomIn size={20} /></button>
            </div>

            {/* Navigation */}
            <button
                onClick={onPrev}
                disabled={currentIndex === 0}
                className="absolute left-4 top-1/2 transform -translate-y-1/2 glass-btn p-3 rounded-full opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-0"
            >
                <ChevronLeft size={24} />
            </button>
            <button
                onClick={onNext}
                disabled={currentIndex === totalImages - 1}
                className="absolute right-4 top-1/2 transform -translate-y-1/2 glass-btn p-3 rounded-full opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-0"
            >
                <ChevronRight size={24} />
            </button>
        </div>
    );
};
