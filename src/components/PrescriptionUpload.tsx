import { useState, useRef } from "react";
import { Upload, X, FileText, Check, Loader2, AlertTriangle, Atom } from "lucide-react";
import { Button } from "@/components/ui/button";
import { analyzePrescription, ScannedMedication } from "@/services/visionService";
import { useToast } from "@/hooks/use-toast";
import { useFamily } from "@/contexts/FamilyContext";
import { addDoc, collection, serverTimestamp, getDocs, query, limit } from "firebase/firestore";
import { db, auth, storage } from "@/lib/firebase";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";

interface PrescriptionUploadProps {
    onSuccess?: () => void;
}

export function PrescriptionUpload({ onSuccess }: PrescriptionUploadProps) {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<ScannedMedication[]>([]);
    const [rawText, setRawText] = useState<string>("");
    const [showResults, setShowResults] = useState(false);

    const API_KEY = "b47e0b7fdcab9c2ce60a094ac63f186166fec0ea";
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { toast } = useToast();
    const { activeMember } = useFamily();

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const f = e.target.files[0];
            setFile(f);
            const reader = new FileReader();
            reader.onloadend = () => setPreview(reader.result as string);
            reader.readAsDataURL(f);
            setShowResults(false);
            setRawText("");
        }
    };

    const handleAnalyze = async () => {
        if (!file || !preview) return;
        setLoading(true);
        setRawText("");
        try {
            const result = await analyzePrescription(preview, API_KEY);
            setRawText(result.rawText);
            if (result.medicines.length === 0) {
                toast({ title: "Analysis Complete", description: "No specific medicines matched, but text was extracted.", variant: "destructive" });
            } else {
                setResults(result.medicines);
                toast({ title: "Analysis Complete", description: `Found ${result.medicines.length} medicines.` });
            }
            setShowResults(true);
        } catch (error: unknown) {
            toast({ title: "Analysis Failed", description: "Could not connect to OCR server.", variant: "destructive" });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!auth.currentUser || !activeMember) {
            toast({ title: "Session Error", description: "Please ensure you are logged in.", variant: "destructive" });
            return;
        }

        if (results.length === 0 && !rawText) {
            toast({ title: "Nothing to Save", description: "No data was identified to store.", variant: "destructive" });
            return;
        }

        setLoading(true);
        try {
            const rootPath = activeMember.isPrimary
                ? `users/${auth.currentUser.uid}`
                : `users/${auth.currentUser.uid}/family_members/${activeMember.id}`;

            const medCol = `${rootPath}/medications`;
            const labCol = `${rootPath}/lab_reports`;

            console.log("DEBUG: Attempting save to path:", rootPath);

            const labKeywords = ["glucose", "hba1c", "cholesterol", "tsh", "creatinine", "hemoglobin", "blood", "report", "lab", "test"];
            const isLabReport = labKeywords.some(kw => rawText.toLowerCase().includes(kw));

            // 1. Image Data (Using Base64 instead of Storage)
            let imageUrl = preview || "";

            // 2. Data Write
            if (isLabReport) {
                // Check if this is the first lab report
                const existingLabs = await getDocs(query(collection(db, labCol), limit(1)));
                const isFirst = existingLabs.empty;

                const tests = results.length > 0
                    ? results.map(m => ({
                        name: m.name,
                        value: m.dosage ? m.dosage.replace(/[^0-9.]/g, '') : "0",
                        unit: m.dosage ? m.dosage.replace(/[0-9.]/g, '').trim() : "unit"
                    }))
                    : [{ name: "Pending Review", value: "0", unit: "n/a" }];

                await addDoc(collection(db, labCol), {
                    date: serverTimestamp(),
                    type: "Lab Report",
                    category: "scanned_ocr",
                    tag: isFirst ? "First Report" : "Follow-up",
                    tests,
                    imageUrl,
                    rawText: rawText.slice(0, 1000)
                });
                toast({ title: "Lab Results Saved", description: isFirst ? "First report saved successfully!" : "Follow-up report added to history." });
            } else if (results.length > 0) {
                // Check if these are the first medications
                const existingMeds = await getDocs(query(collection(db, medCol), limit(1)));
                const isFirst = existingMeds.empty;

                const promises = results.map(med => {
                    return addDoc(collection(db, medCol), {
                        name: med.name,
                        dosage: med.dosage || "As directed",
                        frequency: med.frequency || "As directed",
                        description: med.description || "",
                        category: med.category || "General",
                        prescribedAt: serverTimestamp(),
                        dateStr: new Date().toLocaleDateString(), // Human-readable date
                        tag: isFirst ? "Initial Medication" : "Updated Presc.",
                        status: "active",
                        imageUrl,
                        source: "ocr_scan"
                    });
                });
                await Promise.all(promises);
                toast({ title: "Medications Saved", description: isFirst ? "Initial medications saved!" : "Medications added to profile." });
            } else if (rawText) {
                // FALLBACK: Save the raw text as a 'Pending Review' item if no structured data was found
                await addDoc(collection(db, medCol), {
                    name: "New Prescription Entry",
                    dosage: "See image",
                    frequency: "See image",
                    description: rawText.slice(0, 500) + (rawText.length > 500 ? "..." : ""),
                    category: "Manual Review",
                    prescribedAt: serverTimestamp(),
                    dateStr: new Date().toLocaleDateString(),
                    status: "pending",
                    tag: "Manual Review",
                    imageUrl,
                    source: "ocr_scan_raw",
                    rawText: rawText.slice(0, 1000)
                });
                toast({
                    title: "Saved as Note",
                    description: "Manual entry saved for doctor review."
                });
            }

            // Success Reset
            setFile(null);
            setPreview(null);
            setShowResults(false);
            setResults([]);
            setRawText("");
            if (onSuccess) onSuccess();

        } catch (error: unknown) {
            const err = error as any;
            toast({
                title: "Firebase Error",
                description: "Permission denied or database not initialized. Please check your Firebase rules in the console.",
                variant: "destructive"
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="glass-card p-6 animate-fade-in-up">
            <h2 className="text-lg font-display font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-primary" /> Prescription Scanner
            </h2>

            {!showResults ? (
                <div className="space-y-4">
                    <div
                        onClick={() => fileInputRef.current?.click()}
                        className={`border-2 border-dashed border-border/50 rounded-xl p-8 text-center hover:bg-secondary/20 transition-colors cursor-pointer ${preview ? 'border-primary/50 bg-primary/5' : ''}`}
                    >
                        <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
                        {preview ? (
                            <img src={preview} alt="Preview" className="max-h-48 rounded-lg shadow-md mx-auto" />
                        ) : (
                            <div className="space-y-2">
                                <FileText className="w-12 h-12 text-muted-foreground mx-auto" />
                                <p className="text-sm font-medium">Click to upload prescription</p>
                            </div>
                        )}
                    </div>
                    <div className="flex justify-end">
                        <Button onClick={handleAnalyze} disabled={!file || loading} className="glow-indigo">
                            {loading && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                            {loading ? "Analyzing..." : "Analyze Image"}
                        </Button>
                    </div>
                </div>
            ) : (
                <div className="space-y-6">
                    <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold">Identification Results</h3>
                        <Button variant="ghost" size="sm" onClick={() => setShowResults(false)}>Cancel</Button>
                    </div>

                    <div className="bg-secondary/10 rounded-lg border border-border/20 overflow-hidden">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-secondary/30 text-xs uppercase text-muted-foreground">
                                <tr>
                                    <th className="p-3">Item</th>
                                    <th className="p-3">Dose</th>
                                    <th className="p-3">Insights</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/10">
                                {results.map((med, i) => (
                                    <tr key={i} className="hover:bg-white/5">
                                        <td className="p-3 font-medium">{med.name}</td>
                                        <td className="p-3">{med.dosage}</td>
                                        <td className="p-3 text-[10px] text-muted-foreground">{med.category}</td>
                                    </tr>
                                ))}
                                {results.length === 0 && (
                                    <tr><td colSpan={3} className="p-8 text-center italic text-muted-foreground">No matches found.</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>

                    {rawText && (
                        <div className="p-4 rounded-lg bg-black/20 border border-white/5">
                            <h4 className="text-[10px] uppercase text-muted-foreground mb-2">Extracted Text</h4>
                            <p className="text-xs font-mono text-muted-foreground whitespace-pre-wrap">{rawText}</p>
                        </div>
                    )}

                    <div className="flex justify-end gap-3">
                        <Button variant="outline" onClick={() => setShowResults(false)}>Re-Scan</Button>
                        <Button onClick={handleSave} className="glow-indigo" disabled={loading || (results.length === 0 && !rawText)}>
                            {loading && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                            Save to Profile
                        </Button>
                    </div>
                </div>
            )}
        </div>
    );
}
