export type CountMethod = "exact" | "estimate";

export interface TokenCount {
  readonly tokens: number;
  readonly method: CountMethod;
  readonly tokenizer: string;
  readonly errorBandPct: number | null;
}

export interface ContextPart {
  readonly label: string;
  readonly kind: "system" | "file" | "overhead";
  readonly count: TokenCount;
}

export interface BudgetReport {
  readonly model: string;
  readonly windowTokens: number;
  readonly reservedOutputTokens: number;
  readonly parts: readonly ContextPart[];
  readonly inputTokens: number;
  readonly leftForReply: number;
  readonly overBudgetBy: number;
}

export function share(part: ContextPart, report: BudgetReport): number {
  if (report.windowTokens <= 0) return 0;
  return (part.count.tokens / report.windowTokens) * 100;
}

export function isFit(report: BudgetReport): boolean {
  return report.inputTokens + report.reservedOutputTokens <= report.windowTokens;
}

export function worstCase(report: BudgetReport): number {
  return report.parts.reduce((total, part) => {
    const band = part.count.errorBandPct ?? 0;
    return total + part.count.tokens * (1 + band / 100);
  }, 0);
}

export function cutUntilFits(report: BudgetReport, order: readonly string[]): string[] {
  const removed: string[] = [];
  let used = report.inputTokens;
  const byLabel = new Map(report.parts.map((part) => [part.label, part.count.tokens]));
  for (const label of order) {
    if (used + report.reservedOutputTokens <= report.windowTokens) break;
    const tokens = byLabel.get(label);
    if (tokens === undefined) continue;
    used -= tokens;
    removed.push(label);
  }
  return removed;
}
