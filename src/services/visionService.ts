import { getMedicineDetails } from "./medicineService";

export interface ScannedMedication {
    name: string;
    dosage: string;
    frequency: string;
    confidence: number;
    description?: string;
    category?: string;
    uses?: string[];
    side_effects?: string[];
}

const BACKEND_OCR_URL = import.meta.env.VITE_OCR_BACKEND_URL || "http://localhost:8001/upload-prescription";

// Mock data for fallback/demo mode
const MOCK_MEDICATIONS: ScannedMedication[] = [
    { name: "Amoxicillin", dosage: "500mg", frequency: "Twice daily", confidence: 0.95, description: "Antibiotic for bacterial infections", category: "Antibiotics", uses: ["Bacterial infections"], side_effects: ["Nausea"] },
    { name: "Metformin", dosage: "500mg", frequency: "Once daily", confidence: 0.88, description: "Type 2 Diabetes medication", category: "Antidiabetic", uses: ["Diabetes"], side_effects: ["Stomach upset"] }
];

export interface AnalysisResult {
    medicines: ScannedMedication[];
    rawText: string;
}

export const analyzePrescription = async (base64Image: string, apiKey: string): Promise<AnalysisResult> => {
    try {
        const blob = await (await fetch(base64Image)).blob();
        const formData = new FormData();
        formData.append("file", blob, "prescription.png");

        const response = await fetch(BACKEND_OCR_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Backend OCR failed with status ${response.status}`);
        }

        const data = await response.json();

        if (data.status === "error") {
            throw new Error(data.message);
        }

        const enrichedMeds = await Promise.all(
            data.medicines.map(async (med: { name: string; dose: string }) => {
                const details = await getMedicineDetails(med.name);
                return {
                    name: med.name,
                    dosage: med.dose,
                    frequency: "As directed",
                    confidence: 0.9,
                    description: details.uses[0],
                    category: details.category,
                    uses: details.uses,
                    side_effects: details.side_effects
                };
            })
        );

        return {
            medicines: enrichedMeds,
            rawText: data.raw_text
        };

    } catch (error) {
        console.warn("OCR Backend not available or failed.", error);
        return {
            medicines: [], // Return empty so we can see the raw text is failing
            rawText: "ERROR: Could not connect to OCR backend."
        };
    }
};
