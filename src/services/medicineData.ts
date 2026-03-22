export const MEDICINE_DATABASE = [
    // DIABETES
    { name: "Metformin", uses: ["Controls blood sugar", "Type 2 Diabetes"], side_effects: ["Nausea", "Diarrhea"], category: "Antidiabetic" },
    { name: "Glipizide", uses: ["Lower blood sugar", "Type 2 Diabetes"], side_effects: ["Dizziness", "Drowsiness"], category: "Antidiabetic" },
    { name: "Glyburide", uses: ["Lower blood sugar", "Type 2 Diabetes"], side_effects: ["Nausea", "Heartburn"], category: "Antidiabetic" },
    { name: "Pioglitazone", uses: ["Lower blood sugar", "Type 2 Diabetes"], side_effects: ["Weight gain", "Muscle pain"], category: "Antidiabetic" },
    { name: "Sitagliptin", uses: ["Lower blood sugar", "Type 2 Diabetes"], side_effects: ["Upper respiratory infection"], category: "Antidiabetic" },
    { name: "Insulin", uses: ["Replace insulin hormone", "Diabetes"], side_effects: ["Low blood sugar", "Weight gain"], category: "Antidiabetic" },

    // HYPERTENSION (Blood Pressure)
    { name: "Amlodipine", uses: ["Lower high blood pressure", "Angina"], side_effects: ["Swollen ankles", "Fatigue"], category: "Calcium Channel Blocker" },
    { name: "Lisinopril", uses: ["Lower high blood pressure", "Heart failure"], side_effects: ["Dry cough", "Dizziness"], category: "ACE Inhibitor" },
    { name: "Losartan", uses: ["Lower high blood pressure", "Stroke risk"], side_effects: ["Dizziness", "Musculoskeletal pain"], category: "ARB" },
    { name: "Metoprolol", uses: ["Lower high blood pressure", "Heart rhythm"], side_effects: ["Slow heartbeat", "Dizziness"], category: "Beta Blocker" },
    { name: "Hydrochlorothiazide", uses: ["Lower high blood pressure", "Fluid retention"], side_effects: ["Dehydration", "Dizziness"], category: "Diuretic" },
    { name: "Atenolol", uses: ["Lower high blood pressure", "Chest pain"], side_effects: ["Cold hands/feet", "Fatigue"], category: "Beta Blocker" },
    { name: "Furosemide", uses: ["Reduce fluid buildup", "Heart failure"], side_effects: ["Dehydration", "Loss of electrolytes"], category: "Diuretic" },
    { name: "Ramipril", uses: ["Lower high blood pressure", "Prevent heart attack"], side_effects: ["Cough", "Dizziness"], category: "ACE Inhibitor" },

    // CHOLESTEROL
    { name: "Atorvastatin", uses: ["Lower bad cholesterol", "Prevent heart disease"], side_effects: ["Muscle pain", "Nausea"], category: "Statin" },
    { name: "Simvastatin", uses: ["Lower cholesterol", "Prevent heart disease"], side_effects: ["Constipation", "Stomach pain"], category: "Statin" },
    { name: "Rosuvastatin", uses: ["Lower cholesterol", "Crestor"], side_effects: ["Headache", "Abdominal pain"], category: "Statin" },
    { name: "Pravastatin", uses: ["Lower cholesterol", "Pravachol"], side_effects: ["Muscle pain", "Dizziness"], category: "Statin" },

    // PAIN & INFLAMMATION
    { name: "Paracetamol", uses: ["Pain relief", "Fever reduction"], side_effects: ["Rare", "Liver damage (high dose)"], category: "Analgesic" },
    { name: "Ibuprofen", uses: ["Reduce inflammation", "Pain relief"], side_effects: ["Stomach upset", "Nausea"], category: "NSAID" },
    { name: "Naproxen", uses: ["Reduce inflammation", "Pain relief"], side_effects: ["Heartburn", "Dizziness"], category: "NSAID" },
    { name: "Diclofenac", uses: ["Reduce inflammation", "Joint pain"], side_effects: ["Stomach pain", "Gas"], category: "NSAID" },
    { name: "Aspirin", uses: ["Pain relief", "Fever", "Blood thinner"], side_effects: ["Upset stomach", "Bleeding"], category: "NSAID" },
    { name: "Celecoxib", uses: ["Joint pain", "Arthritis"], side_effects: ["Gas", "Sore throat"], category: "NSAID" },
    { name: "Tramadol", uses: ["Moderate to severe pain"], side_effects: ["Constipation", "Nausea", "Dizziness"], category: "Opioid Analgesic" },

    // ANTIBIOTICS
    { name: "Amoxicillin", uses: ["Bacterial infections", "Chest infections"], side_effects: ["Nausea", "Diarrhea", "Rash"], category: "Antibiotic" },
    { name: "Azithromycin", uses: ["Bacterial infections", "Respiratory"], side_effects: ["Diarrhea", "Stomach cramps"], category: "Antibiotic" },
    { name: "Ciprofloxacin", uses: ["Bacterial infections", "UTI"], side_effects: ["Nausea", "Diarrhea"], category: "Antibiotic" },
    { name: "Cephalexin", uses: ["Bacterial infections", "Skin infections"], side_effects: ["Diarrhea", "Stomach pain"], category: "Antibiotic" },
    { name: "Doxycycline", uses: ["Bacterial infections", "Acne"], side_effects: ["Sensitivity to light", "Nausea"], category: "Antibiotic" },
    { name: "Metronidazole", uses: ["Bacterial & parasitic infections"], side_effects: ["Metallic taste", "Nausea"], category: "Antibiotic" },
    { name: "Clarithromycin", uses: ["Respiratory infections", "Stomach ulcers"], side_effects: ["Altered sense of taste", "Nausea"], category: "Antibiotic" },

    // RESPIRATORY / ASTHMA
    { name: "Albuterol", uses: ["Relieve asthma symptoms", "Shortness of breath"], side_effects: ["Tremors", "Nervousness"], category: "Bronchodilator" },
    { name: "Salbutamol", uses: ["Asthma relief", "COPD"], side_effects: ["Shakey hands", "Headache"], category: "Bronchodilator" },
    { name: "Montelukast", uses: ["Asthma prevention", "Allergy relief"], side_effects: ["Upper respiratory infection"], category: "Leukotriene Receptor Antagonist" },
    { name: "Fluticasone", uses: ["Nasal allergies", "Asthma prevention"], side_effects: ["Headache", "Sore throat"], category: "Corticosteroid" },
    { name: "Budensonide", uses: ["Asthma", "Inflammatory bowel disease"], side_effects: ["Cough", "Hoarseness"], category: "Corticosteroid" },

    // ACID REFLUX / GASTRO
    { name: "Omeprazole", uses: ["Reduce stomach acid", "Heartburn"], side_effects: ["Headache", "Abdominal pain"], category: "PPI" },
    { name: "Pantoprazole", uses: ["GERD", "Stomach acid"], side_effects: ["Diarrhea", "Fatigue"], category: "PPI" },
    { name: "Esomeprazole", uses: ["Nexium", "Acid reflux"], side_effects: ["Dry mouth", "Abdominal pain"], category: "PPI" },
    { name: "Ranitidine", uses: ["Zantac", "Acid relief"], side_effects: ["Headache", "Constipation"], category: "H2 Blocker" },
    { name: "Famotidine", uses: ["Pepcid", "Acid relief"], side_effects: ["Dizziness", "Diarrhea"], category: "H2 Blocker" },

    // THYROID
    { name: "Levothyroxine", uses: ["Underactive thyroid", "Hypothyroidism"], side_effects: ["Palpitations", "Weight loss"], category: "Thyroid Hormone" },

    // ALLERGY / ANTIHISTAMINE
    { name: "Cetirizine", uses: ["Zyrtec", "Common allergies"], side_effects: ["Drowsiness", "Dry mouth"], category: "Antihistamine" },
    { name: "Loratadine", uses: ["Claritin", "Seasonal allergies"], side_effects: ["Headache", "Dry mouth"], category: "Antihistamine" },
    { name: "Fexofenadine", uses: ["Allegra", "Allergy symptoms"], side_effects: ["Dizziness", "Sleepiness"], category: "Antihistamine" },
    { name: "Diphenhydramine", uses: ["Benadryl", "Allergies", "Sleep aid"], side_effects: ["Significant drowsiness"], category: "Antihistamine" },

    // MENTAL HEALTH / NEUROLOGY
    { name: "Sertraline", uses: ["Zoloft", "Depression", "Anxiety"], side_effects: ["Nausea", "Insomnia"], category: "SSRI" },
    { name: "Serta", uses: ["Brand of Sertraline", "Depression", "Anxiety"], side_effects: ["Nausea", "Insomnia"], category: "SSRI" },
    { name: "Escitalopram", uses: ["Lexapro", "Depression", "Anxiety"], side_effects: ["Sweating", "Loss of appetite"], category: "SSRI" },
    { name: "Fluoxetine", uses: ["Prozac", "OCD", "Depression"], side_effects: ["Anxiety", "Nervousness"], category: "SSRI" },
    { name: "Alprazolam", uses: ["Xanax", "Anxiety", "Panic disorder"], side_effects: ["Drowsiness", "Lightheadedness"], category: "Benzodiazepine" },
    { name: "Diazepam", uses: ["Valium", "Anxiety", "Muscle spasms"], side_effects: ["Fatigue", "Muscle weakness"], category: "Benzodiazepine" },
    { name: "Ativan", uses: ["Lorazepam", "Anxiety", "Seizures", "Sedation"], side_effects: ["Drowsiness", "Dizziness", "Weakness"], category: "Benzodiazepine" },
    { name: "Lorazepam", uses: ["Ativan", "Anxiety", "Sleep issues"], side_effects: ["Drowsiness", "Dizziness"], category: "Benzodiazepine" },
    { name: "Rivolil", uses: ["Brand of Clonazepam", "Seizures", "Panic attacks"], side_effects: ["Drowsiness", "Coordination loss"], category: "Benzodiazepine" },
    { name: "Clonazepam", uses: ["Rivotril", "Seizures", "Anxiety"], side_effects: ["Drowsiness", "Dizziness"], category: "Benzodiazepine" },
    { name: "Gabapentin", uses: ["Nerve pain", "Seizures"], side_effects: ["Dizziness", "Drowsiness"], category: "Anticonvulsant" },
    { name: "Qutipin", uses: ["Quetiapine", "Schizophrenia", "Bipolar disorder"], side_effects: ["Weight gain", "Drowsiness"], category: "Antipsychotic" },
    { name: "Quetiapine", uses: ["Seroquel", "Mood stabilizer"], side_effects: ["Drowsiness", "Dry mouth"], category: "Antipsychotic" },
    { name: "Sizodon Plus", uses: ["Risperidone + Trihexyphenidyl", "Schizophrenia", "Psychosis"], side_effects: ["Dry mouth", "Drowsiness", "Restlessness"], category: "Antipsychotic" },

    // VITAMINS / SUPPLEMENTS
    { name: "Calciferol", uses: ["Vitamin D supplement", "Bone health"], side_effects: ["Rare"], category: "Supplement" },
    { name: "Cyanocobalamin", uses: ["Vitamin B12 supplement", "Energy"], side_effects: ["Rare"], category: "Supplement" },
    { name: "Folate", uses: ["Vitamin B9", "Cell production"], side_effects: ["Rare"], category: "Supplement" },
    { name: "Iron", uses: ["Anemia", "Iron deficiency"], side_effects: ["Constipation", "Dark stools"], category: "Supplement" }
];
