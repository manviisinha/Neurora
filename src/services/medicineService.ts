import { db } from "@/lib/firebase";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { MEDICINE_DATABASE } from "./medicineData";

export interface MedicineDetails {
    name: string;
    uses: string[];
    side_effects: string[];
    category: string;
}

const COLLECTION_NAME = "medicine_master";

const capitalize = (str: string) => str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();

const OPENFDA_API_URL = "https://api.fda.gov/drug/label.json";

/**
 * Fetches medicine details from OpenFDA API with fallback to local database.
 */
export const getMedicineDetails = async (name: string): Promise<MedicineDetails> => {
    const inputName = name.trim().toLowerCase();

    // 1. Query OpenFDA API for high-fidelity data on thousands of meds
    try {
        const response = await fetch(`${OPENFDA_API_URL}?search=openfda.brand_name:"${name}"+openfda.generic_name:"${name}"&limit=1`);
        if (response.ok) {
            const data = await response.json();
            if (data.results && data.results.length > 0) {
                const result = data.results[0];
                return {
                    name: result.openfda.brand_name ? result.openfda.brand_name[0] : name,
                    uses: result.indications_and_usage ? [result.indications_and_usage[0].slice(0, 300) + "..."] : ["FDA Approved Medication"],
                    side_effects: result.adverse_reactions ? [result.adverse_reactions[0].slice(0, 300) + "..."] : ["Consult your physician"],
                    category: result.openfda.pharm_class_epc ? result.openfda.pharm_class_epc[0] : "Prescription Medication"
                };
            }
        }
    } catch (apiError) {
        console.warn(`OpenFDA API lookup failed for ${name}:`, apiError);
    }

    // 2. Fuzzy/Partial Match in Local Database (Fallback)
    const logicalMatch = MEDICINE_DATABASE.find(m => {
        const dbName = m.name.toLowerCase();
        return inputName.includes(dbName) || dbName.includes(inputName);
    });

    if (logicalMatch) {
        return logicalMatch;
    }

    // 3. Last Resort Fallback
    return {
        name: name,
        uses: ["Information not found in database"],
        side_effects: [],
        category: "Unknown"
    };
};

/**
 * Seeds Firestore with the expanded medicine list.
 */
export const seedMedicineDatabase = async () => {
    console.log("Starting medicine database seeding...");
    const batchPromises = MEDICINE_DATABASE.map(med => {
        return setDoc(doc(db, COLLECTION_NAME, med.name), med);
    });

    await Promise.all(batchPromises);
    console.log(`Medicine database seeded with ${MEDICINE_DATABASE.length} entries!`);
};
