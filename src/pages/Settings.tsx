import { useState, useEffect } from "react";
import { User, Save, Bell, Shield, Key } from "lucide-react";
import { auth, db } from "@/lib/firebase";
import { doc, getDoc, updateDoc } from "firebase/firestore";
import { onAuthStateChanged } from "firebase/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Separator } from "@/components/ui/separator";

const BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];

interface SettingsData {
    full_name?: string;
    phone?: string;
    date_of_birth?: string;
    blood_group?: string;
    emergency_contact?: string;
    medical_conditions?: string;
    allergies?: string;
    [key: string]: any;
}

export default function Settings() {
    const [loading, setLoading] = useState(true);
    const [formData, setFormData] = useState<SettingsData>({});
    const { toast } = useToast();

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (user) => {
            if (user) {
                try {
                    const docRef = doc(db, "users", user.uid);
                    const docSnap = await getDoc(docRef);
                    if (docSnap.exists()) {
                        setFormData(docSnap.data() as SettingsData);
                    }
                } catch (error) {
                    console.error("Error fetching user data:", error);
                }
            }
            setLoading(false);
        });
        return () => unsubscribe();
    }, []);

    const handleInputChange = (field: string, value: string) => {
        setFormData((prev) => ({ ...prev, [field]: value }));
    };

    const handleSaveProfile = async () => {
        if (!auth.currentUser) return;
        try {
            const docRef = doc(db, "users", auth.currentUser.uid);
            await updateDoc(docRef, formData as any);
            toast({ title: "Settings Saved", description: "Your profile details have been updated." });
        } catch (error: unknown) {
            const err = error as any;
            console.error("Error updating profile:", error);
            toast({ title: "Update Failed", description: err.message, variant: "destructive" });
        }
    };

    if (loading) {
        return <div className="p-8 text-center text-muted-foreground">Loading settings...</div>;
    }

    return (
        <div className="max-w-4xl mx-auto space-y-8 animate-fade-in-up">
            <div>
                <h1 className="text-2xl font-display font-bold text-foreground">Settings</h1>
                <p className="text-sm text-muted-foreground mt-1">Manage your account and preferences</p>
            </div>

            <div className="grid gap-8">
                {/* Profile Section */}
                <div className="glass-card p-6 space-y-6">
                    <div className="flex items-center gap-2 mb-4">
                        <div className="p-2 rounded-lg bg-primary/10">
                            <User className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                            <h2 className="text-lg font-display font-semibold">Personal Information</h2>
                            <p className="text-xs text-muted-foreground">Update your personal details here.</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-1.5">
                            <Label>Full Name</Label>
                            <Input value={formData.full_name || ""} onChange={(e) => handleInputChange("full_name", e.target.value)} />
                        </div>
                        <div className="space-y-1.5">
                            <Label>Phone Number</Label>
                            <Input value={formData.phone || ""} onChange={(e) => handleInputChange("phone", e.target.value)} />
                        </div>
                        <div className="space-y-1.5">
                            <Label>Date of Birth</Label>
                            <Input type="date" value={formData.date_of_birth || ""} onChange={(e) => handleInputChange("date_of_birth", e.target.value)} />
                        </div>
                        <div className="space-y-1.5">
                            <Label>Blood Group</Label>
                            <select
                                value={formData.blood_group || ""}
                                onChange={(e) => handleInputChange("blood_group", e.target.value)}
                                className="w-full h-10 rounded-md bg-secondary/20 border border-border/40 px-3 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 transition-all focus:bg-background"
                            >
                                <option value="">Select</option>
                                {BLOOD_GROUPS.map((bg) => <option key={bg} value={bg}>{bg}</option>)}
                            </select>
                        </div>
                    </div>

                    <Separator className="bg-border/40" />

                    <div className="space-y-4">
                        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Medical Details</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-1.5 md:col-span-2">
                                <Label>Emergency Contact</Label>
                                <Input value={formData.emergency_contact || ""} onChange={(e) => handleInputChange("emergency_contact", e.target.value)} placeholder="Name & Phone" />
                            </div>
                            <div className="space-y-1.5 md:col-span-2">
                                <Label>Medical Conditions</Label>
                                <Input value={formData.medical_conditions || ""} onChange={(e) => handleInputChange("medical_conditions", e.target.value)} placeholder="Diabetes, Hypertension..." />
                            </div>
                            <div className="space-y-1.5 md:col-span-2">
                                <Label>Allergies</Label>
                                <Input value={formData.allergies || ""} onChange={(e) => handleInputChange("allergies", e.target.value)} placeholder="Peanuts, Penicillin..." />
                            </div>
                        </div>
                    </div>

                    <div className="flex justify-end pt-4">
                        <Button onClick={handleSaveProfile} className="glow-indigo gap-2">
                            <Save className="w-4 h-4" /> Save Changes
                        </Button>
                    </div>
                </div>

                {/* Placeholder for other settings */}
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="glass-card p-6 opacity-60 pointer-events-none">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="p-2 rounded-lg bg-neurora-gold/10">
                                <Bell className="w-5 h-5 text-neurora-gold" />
                            </div>
                            <h2 className="text-lg font-display font-semibold">Notifications</h2>
                        </div>
                        <p className="text-sm text-muted-foreground">Manage your alert preferences (Coming Soon).</p>
                    </div>
                    <div className="glass-card p-6 opacity-60 pointer-events-none">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="p-2 rounded-lg bg-neurora-mint/10">
                                <Shield className="w-5 h-5 text-neurora-mint" />
                            </div>
                            <h2 className="text-lg font-display font-semibold">Security</h2>
                        </div>
                        <p className="text-sm text-muted-foreground">Password and security settings (Coming Soon).</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
