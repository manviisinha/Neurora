import { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { collection, query, orderBy, onSnapshot, limit } from "firebase/firestore";
import { db, auth } from "@/lib/firebase";
import { useFamily } from "@/contexts/FamilyContext";

interface LabResult {
  name: string;
  current: string;
  previous: string;
  trend: "improved" | "degraded" | "stable" | "watch";
  unit: string;
}

const timelineData = [
  { month: "Jul", glucose: 115, hba1c: 7.1, cholesterol: 210 },
  { month: "Aug", glucose: 112, hba1c: 7.0, cholesterol: 205 },
  { month: "Sep", glucose: 108, hba1c: 6.9, cholesterol: 200 },
  { month: "Oct", glucose: 105, hba1c: 6.7, cholesterol: 198 },
  { month: "Nov", glucose: 100, hba1c: 6.5, cholesterol: 200 },
  { month: "Dec", glucose: 98, hba1c: 6.3, cholesterol: 203 },
  { month: "Jan", glucose: 95, hba1c: 6.2, cholesterol: 205 },
];

const trendConfig = {
  improved: { icon: TrendingUp, color: "text-neurora-mint", bg: "bg-neurora-mint/10" },
  degraded: { icon: TrendingDown, color: "text-destructive", bg: "bg-destructive/10" },
  watch: { icon: TrendingUp, color: "text-neurora-amber", bg: "bg-neurora-amber/10" },
  stable: { icon: Minus, color: "text-muted-foreground", bg: "bg-muted/30" },
};

const insights = [
  "Your blood glucose levels have improved significantly over the past 6 months, dropping from 115 to 95 mg/dL.",
  "HbA1c shows consistent improvement — you're now within the normal range. Keep up your current regimen.",
  "Cholesterol has risen slightly. Consider dietary adjustments and discuss with your physician at the next visit.",
];

export default function LabAnalytics() {
  const [results, setResults] = useState<LabResult[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { activeMember } = useFamily();

  useEffect(() => {
    if (!auth.currentUser || !activeMember) return;

    let collectionPath = "";
    if (activeMember.isPrimary) {
      collectionPath = `users/${auth.currentUser.uid}/lab_reports`;
    } else {
      collectionPath = `users/${auth.currentUser.uid}/family_members/${activeMember.id}/lab_reports`;
    }

    const q = query(collection(db, collectionPath), orderBy("date", "asc"));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      interface LabReport {
        id: string;
        date: any;
        tests?: { name: string; value: string; unit: string }[];
        [key: string]: any;
      }
      const reports = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() })) as LabReport[];

      if (reports.length === 0) {
        setResults([]);
        setChartData([]);
        setLoading(false);
        return;
      }

      // 1. Process Timeline Chart Data (Chronological)
      const dynamicChartData = reports.map(r => {
        const date = r.date?.toDate ? r.date.toDate() : new Date();
        const month = date.toLocaleString('default', { month: 'short' });
        const entry: any = { month, fullDate: date.toLocaleDateString() };

        r.tests?.forEach(t => {
          if (!t.value) return;
          const val = parseFloat(t.value.replace(/[^0-9.]/g, ''));
          if (!isNaN(val)) {
            const lowerName = t.name.toLowerCase();
            if (lowerName.includes("glucose")) {
              entry["glucose"] = val;
            } else if (lowerName.includes("cholesterol")) {
              entry["cholesterol"] = val;
            } else {
              entry[lowerName] = val;
            }
          }
        });
        return entry;
      });
      setChartData(dynamicChartData);

      // 2. Process Current Results & Trends (Compare newest vs second newest)
      // Helper to normalize test names for grouping
      const normalizeName = (name: string) => {
        const n = name.toLowerCase();
        if (n.includes("glucose") || n.includes("sugar")) return "Glucose";
        if (n.includes("hba1c") || n.includes("a1c")) return "HbA1c";
        if (n.includes("cholesterol")) return "Cholesterol";
        if (n.includes("creatinine")) return "Creatinine";
        if (n.includes("hemoglobin") && !n.includes("a1c")) return "Hemoglobin";
        if (n.includes("tsh") || n.includes("thyroid")) return "TSH";
        return name; // Fallback
      };

      // Group all tests by normalized name
      const grouped = new Map<string, { value: string; unit: string; date: any }[]>();

      reports.forEach(r => {
        r.tests?.forEach(t => {
          if (!t.value) return;
          const key = normalizeName(t.name);
          if (!grouped.has(key)) {
            grouped.set(key, []);
          }
          grouped.get(key)?.push({
            value: t.value,
            unit: t.unit,
            date: r.date
          });
        });
      });

      const dynamicResults: LabResult[] = [];

      grouped.forEach((groupDocs, key) => {
        // Sort by date descending (Newest first)
        groupDocs.sort((a, b) => {
          const dA = a.date?.toDate ? a.date.toDate() : new Date(a.date || 0);
          const dB = b.date?.toDate ? b.date.toDate() : new Date(b.date || 0);
          return dB.getTime() - dA.getTime();
        });

        if (groupDocs.length > 0) {
          const current = groupDocs[0];
          const previous = groupDocs[1]; // might be undefined

          let trend: LabResult["trend"] = "stable";
          if (previous) {
            const cVal = parseFloat(current.value.replace(/[^0-9.]/g, ''));
            const pVal = parseFloat(previous.value.replace(/[^0-9.]/g, ''));
            const isLowerBetter = ["Glucose", "HbA1c", "Cholesterol", "Creatinine", "TSH"].includes(key);

            if (!isNaN(cVal) && !isNaN(pVal)) {
              if (cVal === pVal) trend = "stable";
              else if (isLowerBetter) {
                trend = cVal < pVal ? "improved" : "degraded";
              } else {
                // Higher is better (e.g. Hemoglobin sometimes? usually depends, but let's assume standard logic)
                trend = cVal > pVal ? "improved" : "degraded";
              }
            }
          }

          dynamicResults.push({
            name: key, // Use the normalized name as display
            current: `${current.value} ${current.unit}`,
            previous: previous ? `${previous.value} ${previous.unit}` : "N/A",
            trend: trend,
            unit: current.unit
          });
        }
      });

      // Sort results to put common tests first if desired, or just alphabetically
      // dynamicResults.sort((a, b) => a.name.localeCompare(b.name));

      setResults(dynamicResults);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [activeMember]);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div>
        <h1 className="text-2xl font-display font-bold text-foreground">Lab Analytics</h1>
        <p className="text-sm text-muted-foreground mt-1">Track your lab results over time</p>
      </div>

      {/* Wellness Snapshot */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-display font-semibold text-muted-foreground uppercase tracking-wider mb-4">Wellness Snapshot</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-secondary/20 p-4 rounded-xl border border-border/20">
            <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Latest Report</p>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold font-display">
                {chartData.length > 0 ? chartData[chartData.length - 1].fullDate : "N/A"}
              </span>
            </div>
          </div>
          <div className="bg-secondary/20 p-4 rounded-xl border border-border/20">
            <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Tests Tracked</p>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold font-display">{results.length}</span>
              <span className="text-xs text-muted-foreground">biomarkers</span>
            </div>
          </div>
          <div className="bg-secondary/20 p-4 rounded-xl border border-border/20">
            <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Wellness Status</p>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold font-display text-neurora-mint">Active Monitoring</span>
            </div>
          </div>
        </div>
      </div>

      {/* Results Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <div className="col-span-full py-12 text-center text-muted-foreground italic">Updating analytics engine...</div>
        ) : results.length > 0 ? (
          results.map((result) => {
            const t = trendConfig[result.trend as keyof typeof trendConfig] || trendConfig.stable;
            const TrendIcon = t.icon;
            return (
              <div key={result.name} className="glass-card p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-muted-foreground">{result.name}</p>
                  <div className={`w-6 h-6 rounded-md ${t.bg} flex items-center justify-center`}>
                    <TrendIcon className={`w-3.5 h-3.5 ${t.color}`} />
                  </div>
                </div>
                <p className="text-xl font-display font-bold text-foreground">{result.current}</p>
                <p className="text-xs text-muted-foreground mt-1">Previous: {result.previous}</p>
              </div>
            );
          })
        ) : (
          <div className="col-span-full py-12 glass-card text-center">
            <p className="text-sm text-muted-foreground">No historical lab data found to compare.</p>
            <p className="text-[10px] text-muted-foreground mt-1">Trends will appear after your second report upload.</p>
          </div>
        )}
      </div>

      {/* AI Insights */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-display font-semibold text-muted-foreground uppercase tracking-wider mb-4">AI Insights</h2>
        <div className="space-y-3">
          {insights.map((insight, i) => (
            <div key={i} className="p-3 rounded-lg bg-secondary/30 border border-border/20">
              <p className="text-base font-body leading-relaxed text-foreground">{insight}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
