export interface Item {
    name: string;
    quantity: number;
    price: number;
}

export interface ReceiptData {
    merchant_name: string | null;
    receipt_id: string | null;
    tax_id: string | null;
    date: string | null;
    items: Item[];
    subtotal: number;
    discount: number;
    tax: number;
    total: number;
    currency: string;
    payment_method: string;
}

export type AnnotationStatus = 'unlabeled' | 'labeled' | 'skipped';

export interface Annotation {
    image_path: string; // Base64 or URL
    image_name: string;
    annotation_data: Partial<ReceiptData>;
    status: AnnotationStatus;
    timestamp: string | null;
}
