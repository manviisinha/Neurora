export interface Interaction {
    drug: string;
    severity: "high" | "moderate" | "low";
    description: string;
}

export interface DrugInfo {
    name: string;
    description: string;
    category: string;
    interactions: Interaction[];
}

export const DRUG_DATABASE: Record<string, DrugInfo> = {
    "Amoxicillin": {
        name: "Amoxicillin",
        description: "Antibiotic used to treat bacterial infections.",
        category: "Antibiotics",
        interactions: [
            { drug: "Methotrexate", severity: "moderate", description: "May increase toxicity of methotrexate." },
            { drug: "Warfarin", severity: "moderate", description: "May increase bleeding risk." }
        ]
    },
    "Metformin": {
        name: "Metformin",
        description: "First-line medication for type 2 diabetes control.",
        category: "Antidiabetic",
        interactions: [
            { drug: "Lisinopril", severity: "moderate", description: "May increase risk of lactic acidosis (rare) or renal impairment." },
            { drug: "Iodinated Contrast", severity: "high", description: "Risk of acute kidney failure. Stop Metformin before procedure." }
        ]
    },
    "Lisinopril": {
        name: "Lisinopril",
        description: "ACE inhibitor used for high blood pressure and heart failure.",
        category: "Cardiovascular",
        interactions: [
            { drug: "Metformin", severity: "moderate", description: "Monitor renal function." },
            { drug: "Ibuprofen", severity: "moderate", description: "NSAIDs may reduce antihypertensive effect and harm kidneys." },
            { drug: "Potassium Supplements", severity: "high", description: "Risk of hyperkalemia." }
        ]
    },
    "Atorvastatin": {
        name: "Atorvastatin",
        description: "Statin used to lower cholesterol and reduce heart attack risk.",
        category: "Cardiovascular",
        interactions: [
            { drug: "Clarithromycin", severity: "high", description: "Increased risk of muscle toxicity (rhabdomyolysis)." },
            { drug: "Grapefruit Juice", severity: "moderate", description: "Increases atorvastatin levels." }
        ]
    },
    "Ibuprofen": {
        name: "Ibuprofen",
        description: "NSAID used for pain relief, fever, and inflammation.",
        category: "Pain Relief",
        interactions: [
            { drug: "Lisinopril", severity: "moderate", description: "Reduces blood pressure lowering effect." },
            { drug: "Warfarin", severity: "high", description: "Significantly increases bleeding risk." },
            { drug: "Aspirin", severity: "moderate", description: "May reduce heart-protective effect of low-dose aspirin." }
        ]
    },
    "Warfarin": {
        name: "Warfarin",
        description: "Anticoagulant (blood thinner) to prevent clots.",
        category: "Anticoagulant",
        interactions: [
            { drug: "Aspirin", severity: "high", description: "High bleeding risk." },
            { drug: "Ibuprofen", severity: "high", description: "High bleeding risk." },
            { drug: "Amoxicillin", severity: "moderate", description: "May increase INR/bleeding risk." }
        ]
    },
    "Aspirin": {
        name: "Aspirin",
        description: "Antiplatelet used to prevent heart attacks and strokes.",
        category: "Antiplatelet",
        interactions: [
            { drug: "Warfarin", severity: "high", description: "High bleeding risk." },
            { drug: "Ibuprofen", severity: "moderate", description: "Ibuprofen may block Aspirin's heart benefits." }
        ]
    }
};

export const enrichMedicationData = (ocrName: string): DrugInfo | null => {
    const normalized = ocrName.toLowerCase().trim();
    const match = Object.values(DRUG_DATABASE).find(d =>
        normalized.includes(d.name.toLowerCase()) || d.name.toLowerCase().includes(normalized)
    );
    return match || null;
};
