'use client';

import { createContext, useContext, ReactNode, useState, useCallback } from 'react';

interface ToastContext {
  showToast: (message: string, type: 'info' | 'error' | 'success') => void;
}

const ToastContext = createContext<ToastContext | undefined>(undefined);

interface ToastState {
  message: string;
  type: 'info' | 'error' | 'success';
  visible: boolean;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<ToastState>({
    message: '',
    type: 'info',
    visible: false,
  });

  const showToast = useCallback((message: string, type: 'info' | 'error' | 'success') => {
    setToast({ message, type, visible: true });
    setTimeout(() => setToast(prev => ({ ...prev, visible: false })), 3000);
  }, []);

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {toast.visible && (
        <div className="fixed bottom-4 right-4 px-4 py-2 border transition-opacity duration-300 bg-[var(--bg)] border-[var(--middle)] text-[var(--text)]">
          <p className="text-sm">{toast.message}</p>
        </div>
      )}
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within ToastProvider');
  }
  return context;
}
