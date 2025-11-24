import React, { useEffect, useState } from 'react';
import { Plus, Trash2, Save, SkipForward } from 'lucide-react';
import type { Annotation, Item } from '../types';

interface EditorProps {
    annotation: Annotation | null;
    onSave: (data: Partial<Annotation['annotation_data']>) => void;
    onSkip: () => void;
}

export const Editor: React.FC<EditorProps> = ({ annotation, onSave, onSkip }) => {
    const [formData, setFormData] = useState<Partial<Annotation['annotation_data']>>({});

    useEffect(() => {
        if (annotation) {
            setFormData(annotation.annotation_data || {});
        } else {
            setFormData({});
        }
    }, [annotation]);

    const handleChange = (field: string, value: any) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    };

    const handleItemChange = (index: number, field: keyof Item, value: any) => {
        const newItems = [...(formData.items || [])];
        newItems[index] = { ...newItems[index], [field]: value };
        setFormData(prev => ({ ...prev, items: newItems }));
    };

    const addItem = () => {
        setFormData(prev => ({
            ...prev,
            items: [...(prev.items || []), { name: '', quantity: 1, price: 0 }]
        }));
    };

    const removeItem = (index: number) => {
        const newItems = [...(formData.items || [])];
        newItems.splice(index, 1);
        setFormData(prev => ({ ...prev, items: newItems }));
    };

    if (!annotation) return <div className="glass-panel w-[450px] p-4 text-center text-slate-500">No image selected</div>;

    return (
        <div className="glass-panel w-[450px] flex flex-col min-h-0 h-full">
            <div className="p-4 border-b border-white/10 flex justify-between items-center shrink-0">
                <h2 className="font-semibold text-white">Receipt Data</h2>
                <span className={`px-2 py-1 rounded-full text-xs font-medium border ${annotation.status === 'labeled' ? 'bg-green-900/50 text-green-300 border-green-700' :
                    annotation.status === 'skipped' ? 'bg-red-900/50 text-red-300 border-red-700' :
                        'bg-slate-700 text-slate-300 border-slate-600'
                    }`}>
                    {annotation.status.charAt(0).toUpperCase() + annotation.status.slice(1)}
                </span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                {/* Merchant Info */}
                <div className="space-y-3">
                    <h3 className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Merchant Info</h3>
                    <div className="grid grid-cols-1 gap-3">
                        <div>
                            <label className="text-xs text-slate-400 block mb-1">Merchant Name</label>
                            <input
                                type="text"
                                className="glass-input w-full rounded-lg px-3 py-2 text-sm"
                                value={formData.merchant_name || ''}
                                onChange={e => handleChange('merchant_name', e.target.value)}
                                placeholder="e.g. 7-Eleven"
                            />
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="text-xs text-slate-400 block mb-1">Date</label>
                                <input
                                    type="date"
                                    className="glass-input w-full rounded-lg px-3 py-2 text-sm"
                                    value={formData.date || ''}
                                    onChange={e => handleChange('date', e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="text-xs text-slate-400 block mb-1">Receipt ID</label>
                                <input
                                    type="text"
                                    className="glass-input w-full rounded-lg px-3 py-2 text-sm"
                                    value={formData.receipt_id || ''}
                                    onChange={e => handleChange('receipt_id', e.target.value)}
                                    placeholder="#12345"
                                />
                            </div>
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 block mb-1">Tax ID</label>
                            <input
                                type="text"
                                className="glass-input w-full rounded-lg px-3 py-2 text-sm"
                                value={formData.tax_id || ''}
                                onChange={e => handleChange('tax_id', e.target.value)}
                                placeholder="Tax ID"
                            />
                        </div>
                    </div>
                </div>

                <div className="h-[1px] bg-white/10"></div>

                {/* Items */}
                <div className="space-y-3">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Items</h3>
                        <button onClick={addItem} className="text-xs text-cyan-400 hover:text-cyan-300 font-medium flex items-center gap-1">
                            <Plus size={12} /> Add Item
                        </button>
                    </div>
                    <div className="space-y-2">
                        {formData.items?.map((item, idx) => (
                            <div key={idx} className="grid grid-cols-[2fr_1fr_1fr_auto] gap-2 items-center animate-fade-in">
                                <input
                                    type="text"
                                    className="glass-input w-full rounded-lg px-2 py-1.5 text-xs"
                                    placeholder="Item"
                                    value={item.name}
                                    onChange={e => handleItemChange(idx, 'name', e.target.value)}
                                />
                                <input
                                    type="number"
                                    className="glass-input w-full rounded-lg px-2 py-1.5 text-xs"
                                    placeholder="Qty"
                                    value={item.quantity}
                                    onChange={e => handleItemChange(idx, 'quantity', parseFloat(e.target.value))}
                                />
                                <input
                                    type="number"
                                    className="glass-input w-full rounded-lg px-2 py-1.5 text-xs"
                                    placeholder="Price"
                                    value={item.price}
                                    onChange={e => handleItemChange(idx, 'price', parseFloat(e.target.value))}
                                />
                                <button onClick={() => removeItem(idx)} className="text-slate-500 hover:text-red-400">
                                    <Trash2 size={14} />
                                </button>
                            </div>
                        ))}
                        {(!formData.items || formData.items.length === 0) && (
                            <div className="text-center text-xs text-slate-600 py-2">No items added</div>
                        )}
                    </div>
                </div>

                <div className="h-[1px] bg-white/10"></div>

                {/* Totals */}
                <div className="space-y-3">
                    <h3 className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Totals</h3>
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="text-xs text-slate-400 block mb-1">Subtotal</label>
                            <input
                                type="number"
                                className="glass-input w-full rounded-lg px-3 py-2 text-sm text-right"
                                value={formData.subtotal || ''}
                                onChange={e => handleChange('subtotal', parseFloat(e.target.value))}
                            />
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 block mb-1">Discount</label>
                            <input
                                type="number"
                                className="glass-input w-full rounded-lg px-3 py-2 text-sm text-right text-red-400"
                                value={formData.discount || ''}
                                onChange={e => handleChange('discount', parseFloat(e.target.value))}
                            />
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 block mb-1">Tax</label>
                            <input
                                type="number"
                                className="glass-input w-full rounded-lg px-3 py-2 text-sm text-right"
                                value={formData.tax || ''}
                                onChange={e => handleChange('tax', parseFloat(e.target.value))}
                            />
                        </div>
                        <div>
                            <label className="text-xs text-slate-400 block mb-1">Total</label>
                            <input
                                type="number"
                                className="glass-input w-full rounded-lg px-3 py-2 text-sm text-right font-bold text-cyan-400"
                                value={formData.total || ''}
                                onChange={e => handleChange('total', parseFloat(e.target.value))}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Footer Actions */}
            <div className="p-4 border-t border-white/10 flex gap-3 shrink-0">
                <button onClick={onSkip} className="glass-btn flex-1 py-2 rounded-lg text-sm font-medium hover:bg-red-500/20 hover:border-red-500/50 hover:text-red-200 flex items-center justify-center gap-2">
                    <SkipForward size={16} /> Skip
                </button>
                <button onClick={() => onSave(formData)} className="btn-primary flex-[2] py-2 rounded-lg text-sm font-medium shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2">
                    <Save size={16} /> Save & Next
                </button>
            </div>
        </div>
    );
};
