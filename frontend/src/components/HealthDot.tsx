export function HealthDot({ status }: { status: string }) {
  const color =
    status === "ok"
      ? "bg-emerald-400"
      : status === "degraded"
      ? "bg-amber-400"
      : status === "down"
      ? "bg-rose-500"
      : "bg-slate-500";
  return (
    <span className="relative flex h-2.5 w-2.5">
      {status === "ok" && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400/60" />
      )}
      <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${color}`} />
    </span>
  );
}
