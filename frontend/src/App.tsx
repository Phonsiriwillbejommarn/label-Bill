import { useState } from 'react';
import axios from 'axios';
import { Download, Zap, FolderOpen } from 'lucide-react';
import { ImageViewer } from './components/ImageViewer';
import { Editor } from './components/Editor';
import type { Annotation } from './types';

function App() {
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [model, setModel] = useState('typhoon-hf');

  const currentAnnotation = annotations[currentIndex] || null;

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []).filter(f => f.type.startsWith('image/'));
    if (files.length === 0) return;

    const newAnnotations: Annotation[] = [];
    let loaded = 0;

    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (ev.target?.result) {
          newAnnotations.push({
            image_path: ev.target.result as string,
            image_name: file.name,
            annotation_data: {},
            status: 'unlabeled',
            timestamp: null
          });
          loaded++;
          if (loaded === files.length) {
            setAnnotations(prev => [...prev, ...newAnnotations]);
          }
        }
      };
      reader.readAsDataURL(file);
    });
    e.target.value = '';
  };

  const handleProcess = async () => {
    if (isProcessing) return;
    setIsProcessing(true);

    const newAnnotations = [...annotations];

    for (let i = 0; i < newAnnotations.length; i++) {
      if (newAnnotations[i].status === 'labeled') continue;

      try {
        const base64 = newAnnotations[i].image_path.split(',')[1];
        let data;

        if (model === 'typhoon-hf') {
          const res = await axios.post('http://localhost:5001/api/process-typhoon', { image: base64 });
          data = res.data.data;
        } else {
          // Ollama logic
          const prompt = `Extract receipt data as JSON: { "merchant_name": string, "date": string, "receipt_id": string, "items": [{ "name": string, "quantity": number, "price": number }], "total": number, "tax": number }`;
          const res = await axios.post('http://localhost:11434/api/generate', {
            model: model,
            prompt: prompt,
            images: [base64],
            stream: false,
            format: "json"
          });
          data = JSON.parse(res.data.response);
        }

        if (data) {
          // Helper to parse numbers from strings like "1,200.00"
          const parseNum = (val: any) => {
            if (typeof val === 'number') return val;
            if (typeof val === 'string') return parseFloat(val.replace(/,/g, '')) || 0;
            return 0;
          };

          const sanitizedData = {
            ...data,
            subtotal: parseNum(data.subtotal),
            discount: parseNum(data.discount),
            tax: parseNum(data.tax),
            total: parseNum(data.total),
            items: (data.items || []).map((item: any) => ({
              name: item.name,
              quantity: parseNum(item.quantity),
              price: parseNum(item.price)
            }))
          };

          // FIX: Create a NEW object to trigger React re-render and useEffect in child components
          newAnnotations[i] = {
            ...newAnnotations[i],
            annotation_data: { ...newAnnotations[i].annotation_data, ...sanitizedData },
            status: 'labeled'
          };

          console.log("Updated annotation:", newAnnotations[i]); // Debug log
          setAnnotations([...newAnnotations]); // Update state progressively
        }
      } catch (err) {
        console.error("Error processing image:", err);
      }
    }
    setIsProcessing(false);
  };

  const handleSave = (data: any) => {
    const newAnnotations = [...annotations];
    newAnnotations[currentIndex].annotation_data = data;
    newAnnotations[currentIndex].status = 'labeled';
    setAnnotations(newAnnotations);

    if (currentIndex < annotations.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
  };

  const handleSkip = () => {
    const newAnnotations = [...annotations];
    newAnnotations[currentIndex].status = 'skipped';
    setAnnotations(newAnnotations);

    if (currentIndex < annotations.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
  };

  const handleExport = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(annotations, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "receipt_labels.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  // Stats
  const total = annotations.length;
  const labeled = annotations.filter(a => a.status === 'labeled').length;
  const skipped = annotations.filter(a => a.status === 'skipped').length;
  const progress = total ? Math.round(((labeled + skipped) / total) * 100) : 0;

  return (
    <div className="h-screen flex flex-col p-4 gap-4">
      {/* Header */}
      <header className="glass-panel p-4 flex justify-between items-center shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-400 to-cyan-300 flex items-center justify-center text-slate-900 font-bold text-xl shadow-lg shadow-cyan-500/20">AI</div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-wide">Receipt Labeling Tool</h1>
            <p className="text-xs text-slate-400">Powered by Ollama & Typhoon</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex flex-col">
            <label className="text-[10px] uppercase tracking-wider text-slate-400 mb-1">AI Model</label>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              className="glass-input rounded-lg px-3 py-1.5 text-sm min-w-[200px] bg-slate-800"
            >
              <option value="typhoon-hf">⚡ Typhoon OCR (FastAPI)</option>
              <option value="scb10x/typhoon-ocr1.5-3b">🦙 Ollama: Typhoon-OCR-3B</option>
              <option value="qwen2.5-vl">🦙 Ollama: Qwen2.5-VL</option>
            </select>
          </div>

          <div className="h-8 w-[1px] bg-white/10"></div>

          <label className="glass-btn px-4 py-2 rounded-lg text-sm flex items-center gap-2 cursor-pointer">
            <FolderOpen size={16} /> Open Images
            <input type="file" multiple accept="image/*" onChange={handleFileUpload} className="hidden" />
          </label>

          <button
            onClick={handleProcess}
            disabled={isProcessing || annotations.length === 0}
            className="btn-primary px-6 py-2 rounded-lg text-sm flex items-center gap-2 transition-all"
          >
            {isProcessing ? <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div> : <Zap size={16} />}
            Auto Label
          </button>

          <button
            onClick={handleExport}
            disabled={annotations.length === 0}
            className="glass-btn px-4 py-2 rounded-lg text-sm flex items-center gap-2"
          >
            <Download size={16} /> Export
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex gap-4 min-h-0">
        <ImageViewer
          imageSrc={currentAnnotation?.image_path || null}
          imageName={currentAnnotation?.image_name || null}
          currentIndex={currentIndex}
          totalImages={total}
          onPrev={() => setCurrentIndex(prev => prev - 1)}
          onNext={() => setCurrentIndex(prev => prev + 1)}
        />
        <Editor
          annotation={currentAnnotation}
          onSave={handleSave}
          onSkip={handleSkip}
        />
      </main>

      {/* Footer Stats */}
      <footer className="glass-panel p-3 flex justify-between items-center text-xs text-slate-400 shrink-0">
        <div className="flex gap-6">
          <span>Total: <strong className="text-white">{total}</strong></span>
          <span>Labeled: <strong className="text-cyan-400">{labeled}</strong></span>
          <span>Skipped: <strong className="text-red-400">{skipped}</strong></span>
        </div>
        <div className="flex items-center gap-3 w-1/3">
          <span>Progress</span>
          <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-400 to-cyan-400 transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <span>{progress}%</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
