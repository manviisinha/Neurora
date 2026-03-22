# Neurora Dashboard

Neurora is an advanced, AI-powered health monitoring dashboard designed to simplify complex medical data into actionable insights. It integrates lab report analysis, medication conflict detection, real-time doctor availability, and intelligent conversational assistance.

## 🚀 Key Features

### 1. **MediScanner (OCR Prescription Analysis)**
   - Simply upload an image of your prescription.
   - Using advanced OCR (EasyOCR + OpenCV), the backend extracts medication names, dosages, and instructions.
   - Automatically populates your digital medicine cabinet.

### 2. **Conflict Engine**
   - **Real-time Drug Interaction Checks**: Analyzes your current medication list against a comprehensive drug interaction database.
   - Detects potential conflicts (e.g., "Aspirin + Ibuprofen") and warns about severity levels.
   - Provides clear, actionable advice on managing multiple prescriptions.
   
### 3. **Lab Analytics & Trends**
   - **Smart Trend Analysis**: Upload multiple lab reports (blood tests, lipid profiles, etc.).
   - The system normalizes test names (e.g., grouping "Fasting Sugar" and "Glucose") to track historical trends.
   - Visualizes your progress with dynamic charts:
     - **Wellness Snapshot**: Quick view of latest vitals.
     - **Trend Indicators**: Know instantly if a biomarker is "Stable", "Improving", or "Degrading".

### 4. **Aura AI Assistant**
   - A conversational AI integrated directly into the dashboard.
   - **Context-Aware**: Understands your specific health data.
     - *"What are my current medicines?"*
     - *"Any conflicts in my prescription?"*
     - *"Show my last cholesterol reading."*
   - Provides instant, data-driven answers without generic web searches.

### 5. **Expert Locator (Live Map)**
   - Find specialists near you with our **Live Expert Map**.
   - **Dynamic Filtering**: Filter by specialty (Cardiologist, Dermatologist, etc.) and distance radius.
   - **Real-Time Visualization**: The map automatically adjusts its zoom and focus based on your search criteria.

### 6. **Emergency SOS**
   - A dedicated, always-accessible SOS button.
   - One-tap connection to emergency services (112) with safeguard confirmation.

## 🛠️ Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Recharts, Lucide Icons, Vite
- **Backend**: Python (FastAPI), Uvicorn, EasyOCR, OpenCV, NumPy
- **Database & Auth**: Firebase (Firestore, Authentication)
- **Deployment**: Vercel (Frontend), Railway/Render (Backend recommended)

## 📦 Installation & Setup

### Prerequisites
- Node.js (v18+)
- Python (v3.9+)

### Frontend
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend
```bash
cd server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the OCR server
python main.py
```

## 🔒 Security & Privacy
- **Strict Access Control**: Users must sign up and verify email to access the dashboard.
- **Data Privacy**: All medical data is stored securely in Firestore with user-scoped permissions.

Made By : Manvi Sinha
