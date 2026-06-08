import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { RetrievalMode } from "./types";

export interface Settings {
  // Empty string => use the Vite dev proxy ("/api/v1/...") same-origin.
  baseUrl: string;
  apiKey: string;
  useAgentic: boolean;
  mode: RetrievalMode | "auto";
  topK: number;
  stream: boolean;
  /** 专家模式 (专业研究): 多轮递进式文献检索 + 综述综合。false = 快速检索 */
  professional: boolean;
  /** 目标知识库集合名, 空串 = 使用配置默认集合 */
  collection: string;
}

const DEFAULTS: Settings = {
  baseUrl: "",
  apiKey: "",
  useAgentic: true,
  mode: "auto",
  topK: 5,
  stream: true,
  professional: false,
  collection: "",
};

const STORAGE_KEY = "dp-rag-settings";

interface SettingsContextValue {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
  reset: () => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

function load(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return DEFAULTS;
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(load);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const value = useMemo<SettingsContextValue>(
    () => ({
      settings,
      update: (patch) => setSettings((s) => ({ ...s, ...patch })),
      reset: () => setSettings(DEFAULTS),
    }),
    [settings]
  );

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider");
  return ctx;
}
