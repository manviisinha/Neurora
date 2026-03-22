import { useState } from "react";
import { useNavigate, Link, useSearchParams } from "react-router-dom";
import { auth, db } from "@/lib/firebase";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  sendPasswordResetEmail,
  sendEmailVerification,
  GoogleAuthProvider,
  signInWithPopup
} from "firebase/auth";
import { doc, setDoc, getDoc } from "firebase/firestore";
import { Brain, ArrowRight, Eye, EyeOff, Mail, Lock, User, Phone, Calendar, Droplets, Heart, AlertTriangle, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

type Mode = "login" | "signup" | "forgot-password";

const BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];
const GENDERS = ["Male", "Female", "Other", "Prefer not to say"];

export default function Auth() {
  const [searchParams] = useSearchParams();
  const [mode, setMode] = useState<Mode>(searchParams.get("mode") === "signup" ? "signup" : "login");
  const [step, setStep] = useState(1); // 1 = credentials, 2 = patient details
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  // Credentials
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");

  // Patient details (step 2)
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [gender, setGender] = useState("");
  const [bloodGroup, setBloodGroup] = useState("");
  const [phone, setPhone] = useState("");
  const [emergencyContact, setEmergencyContact] = useState("");
  const [medicalConditions, setMedicalConditions] = useState("");
  const [allergies, setAllergies] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      toast({ title: "Welcome back!", description: "Successfully logged in." });
      navigate("/dashboard");
    } catch (error: unknown) {
      const err = error as any;
      console.error("Login Error:", error);

      let message = err.message;
      if (err.code === "auth/user-not-found" || err.code === "auth/invalid-credential") {
        message = "Account not found or incorrect password. Please sign up if you don't have an account.";
      } else if (err.code === "auth/wrong-password") {
        message = "Incorrect password. Please try again.";
      }

      toast({ title: "Login failed", description: message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setLoading(true);
    const provider = new GoogleAuthProvider();
    try {
      const result = await signInWithPopup(auth, provider);
      const user = result.user;

      // Check if user profile exists
      const docRef = doc(db, "users", user.uid);
      const docSnap = await getDoc(docRef);

      if (mode === "login") {
        if (docSnap.exists()) {
          toast({ title: "Welcome back!", description: `Successfully logged in as ${user.displayName || user.email}` });
          navigate("/dashboard");
        } else {
          // Account doesn't exist, but we are in login mode -> BLOCK
          await auth.signOut();
          toast({ title: "Account not found", description: "Please sign up first to create an account.", variant: "destructive" });
        }
      } else {
        // Signup Mode
        if (docSnap.exists()) {
          // Already exists, just log in
          toast({ title: "Account exists", description: "Logging you in..." });
          navigate("/dashboard");
        } else {
          // New user -> Move to Step 2 to collect details
          setFullName(user.displayName || "");
          setEmail(user.email || "");
          setStep(2);
          toast({ title: "Google Account Verified", description: "Please complete your patient details." });
        }
      }
    } catch (error: unknown) {
      const err = error as any;
      console.error("Google Login Error:", error);
      toast({ title: "Google Login failed", description: err.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      toast({ title: "Email required", description: "Please enter your email address.", variant: "destructive" });
      return;
    }
    setLoading(true);
    try {
      await sendPasswordResetEmail(auth, email);
      toast({ title: "Email sent", description: "Check your inbox for password reset instructions." });
      setMode("login");
    } catch (error: unknown) {
      const err = error as any;
      toast({ title: "Error", description: err.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const handleSignupStep1 = (e: React.FormEvent) => {
    e.preventDefault();
    if (!fullName.trim() || !email.trim() || !password.trim()) {
      toast({ title: "Missing fields", description: "Please fill in all required fields.", variant: "destructive" });
      return;
    }
    if (password.length < 6) {
      toast({ title: "Weak password", description: "Password must be at least 6 characters.", variant: "destructive" });
      return;
    }
    setStep(2);
  };

  const handleSignupStep2 = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      let user;

      if (auth.currentUser) {
        // Case: User already authenticated via Google Search
        user = auth.currentUser;
      } else {
        // Case: Standard Email/Password Signup
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        user = userCredential.user;
        // Send verification email
        await sendEmailVerification(user);
      }

      // Save user profile to Firestore
      await setDoc(doc(db, "users", user.uid), {
        uid: user.uid,
        email: user.email,
        full_name: fullName,
        date_of_birth: dateOfBirth || null,
        gender: gender || null,
        blood_group: bloodGroup || null,
        phone: phone || null,
        emergency_contact: emergencyContact || null,
        medical_conditions: medicalConditions || null,
        allergies: allergies || null,
        created_at: new Date().toISOString()
      }, { merge: true });

      toast({
        title: "Account created!",
        description: auth.currentUser ? "Welcome to Neuralis!" : "Please check your email to verify your account before logging in.",
      });

      if (auth.currentUser) {
        navigate("/dashboard");
      } else {
        setMode("login");
        setStep(1);
      }
    } catch (error: unknown) {
      const err = error as any;
      console.error(error);
      toast({ title: "Signup failed", description: err.message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setStep(1);
    setEmail("");
    setPassword("");
    setFullName("");
    setDateOfBirth("");
    setGender("");
    setBloodGroup("");
    setPhone("");
    setEmergencyContact("");
    setMedicalConditions("");
    setAllergies("");
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6 relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[500px] h-[500px] rounded-full opacity-15" style={{ background: "radial-gradient(circle, hsl(var(--neurora-indigo) / 0.5), transparent 70%)" }} />
      </div>

      <div className="w-full max-w-md relative z-10 animate-fade-in-up">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 justify-center mb-8">
          <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center glow-indigo">
            <Brain className="w-6 h-6 text-primary" />
          </div>
          <span className="font-display text-2xl font-bold text-gradient-indigo tracking-tight">Neurora</span>
        </Link>

        <div className="glass-card p-8">
          {/* Mode tabs with sliding indicator */}
          <div className="relative flex rounded-xl bg-secondary/40 p-1 mb-8">
            <div
              className={`absolute inset-y-1 w-[calc(50%-4px)] rounded-lg bg-background shadow-sm transition-all duration-300 ease-spring ${mode === "login" || mode === "forgot-password" ? "left-1" : "left-[calc(50%)]"
                }`}
            />
            <button
              onClick={() => { setMode("login"); resetForm(); }}
              className={`relative flex-1 py-2.5 text-sm font-display font-medium transition-colors z-10 ${mode === "login" || mode === "forgot-password" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
            >
              Log In
            </button>
            <button
              onClick={() => { setMode("signup"); resetForm(); }}
              className={`relative flex-1 py-2.5 text-sm font-display font-medium transition-colors z-10 ${mode === "signup" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
            >
              Sign Up
            </button>
          </div>

          <div className="relative min-h-[300px]">
            {/* LOGIN */}
            {mode === "login" && (
              <form key="login" onSubmit={handleLogin} className="space-y-5 animate-fade-in-up">
                <div className="space-y-2 group">
                  <Label htmlFor="login-email" className="text-sm font-display text-muted-foreground group-focus-within:text-primary transition-colors">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                    <Input
                      id="login-email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10 bg-secondary/20 border-border/40 focus:bg-background transition-all duration-300"
                      required
                    />
                  </div>
                </div>
                <div className="space-y-2 group">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="login-password" className="text-sm font-display text-muted-foreground group-focus-within:text-primary transition-colors">Password</Label>
                    <button type="button" onClick={() => setMode("forgot-password")} className="text-xs text-primary hover:text-primary/80 transition-colors">Forgot?</button>
                  </div>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                    <Input
                      id="login-password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="pl-10 pr-10 bg-secondary/20 border-border/40 focus:bg-background transition-all duration-300"
                      required
                    />
                    <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors">
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
                <Button type="submit" className="w-full glow-indigo h-11 text-base" size="lg" disabled={loading}>
                  {loading ? "Signing in..." : "Sign In"} <ArrowRight className="w-4 h-4 ml-1" />
                </Button>

                <div className="relative my-4">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t border-border/50" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3">
                  <Button type="button" variant="outline" onClick={handleGoogleLogin} disabled={loading} className="w-full">
                    <svg className="mr-2 h-4 w-4" aria-hidden="true" focusable="false" data-prefix="fab" data-icon="google" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 488 512">
                      <path fill="currentColor" d="M488 261.8C488 403.3 391.1 504 248 504 110.8 504 0 393.2 0 256S110.8 8 248 8c66.8 0 123 24.5 166.3 64.9l-67.5 64.9C258.5 52.6 94.3 116.6 94.3 256c0 86.5 69.1 156.6 153.7 156.6 98.2 0 135-70.4 140.8-106.9H248v-85.3h236.1c2.3 12.7 3.9 24.9 3.9 41.4z"></path>
                    </svg>
                    Google
                  </Button>
                </div>
              </form>
            )}

            {/* FORGOT PASSWORD */}
            {mode === "forgot-password" && (
              <form key="forgot-password" onSubmit={handleForgotPassword} className="space-y-5 animate-fade-in-up">
                <div className="flex items-center gap-2 mb-2">
                  <button type="button" onClick={() => setMode("login")} className="text-muted-foreground hover:text-foreground transition-colors p-1 -ml-1">
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <p className="text-xs text-muted-foreground font-display uppercase tracking-wider">Reset Password</p>
                </div>
                <div className="space-y-2 group">
                  <Label htmlFor="forgot-email" className="text-sm font-display text-muted-foreground group-focus-within:text-primary transition-colors">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                    <Input
                      id="forgot-email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10 bg-secondary/20 border-border/40 focus:bg-background transition-all duration-300"
                      required
                    />
                  </div>
                </div>
                <Button type="submit" className="w-full glow-indigo h-11 text-base" size="lg" disabled={loading}>
                  {loading ? "Sending..." : "Send Reset Link"} <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </form>
            )}

            {/* SIGNUP STEP 1 */}
            {mode === "signup" && step === 1 && (
              <form key="signup-step-1" onSubmit={handleSignupStep1} className="space-y-5 animate-fade-in-up">
                <div className="flex items-center gap-2 mb-2">
                  <div className="h-1 flex-1 bg-primary rounded-full"></div>
                  <div className="h-1 flex-1 bg-secondary rounded-full"></div>
                </div>
                <p className="text-xs text-muted-foreground font-display uppercase tracking-wider mb-4">Account Creation</p>

                <div className="space-y-2 group">
                  <Label htmlFor="signup-name" className="text-sm font-display text-muted-foreground group-focus-within:text-primary transition-colors">Full Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                    <Input
                      id="signup-name"
                      placeholder="John Doe"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      className="pl-10 bg-secondary/20 border-border/40 focus:bg-background transition-all duration-300"
                      required
                    />
                  </div>
                </div>
                <div className="space-y-2 group">
                  <Label htmlFor="signup-email" className="text-sm font-display text-muted-foreground group-focus-within:text-primary transition-colors">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                    <Input
                      id="signup-email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-10 bg-secondary/20 border-border/40 focus:bg-background transition-all duration-300"
                      required
                    />
                  </div>
                </div>
                <div className="space-y-2 group">
                  <Label htmlFor="signup-password" className="text-sm font-display text-muted-foreground group-focus-within:text-primary transition-colors">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                    <Input
                      id="signup-password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Min 6 characters"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="pl-10 pr-10 bg-secondary/20 border-border/40 focus:bg-background transition-all duration-300"
                      required
                      minLength={6}
                    />
                    <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors">
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
                <Button type="submit" className="w-full glow-indigo h-11 text-base" size="lg">
                  Continue <ChevronRight className="w-4 h-4 ml-1" />
                </Button>

                <div className="relative my-4">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t border-border/50" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3">
                  <Button type="button" variant="outline" onClick={handleGoogleLogin} disabled={loading} className="w-full">
                    <svg className="mr-2 h-4 w-4" aria-hidden="true" focusable="false" data-prefix="fab" data-icon="google" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 488 512">
                      <path fill="currentColor" d="M488 261.8C488 403.3 391.1 504 248 504 110.8 504 0 393.2 0 256S110.8 8 248 8c66.8 0 123 24.5 166.3 64.9l-67.5 64.9C258.5 52.6 94.3 116.6 94.3 256c0 86.5 69.1 156.6 153.7 156.6 98.2 0 135-70.4 140.8-106.9H248v-85.3h236.1c2.3 12.7 3.9 24.9 3.9 41.4z"></path>
                    </svg>
                    Google
                  </Button>
                </div>
              </form>
            )}

            {/* SIGNUP STEP 2 — Patient Details */}
            {mode === "signup" && step === 2 && (
              <form key="signup-step-2" onSubmit={handleSignupStep2} className="space-y-5 animate-fade-in-up">
                <div className="flex items-center gap-2 mb-2">
                  <div className="h-1 flex-1 bg-primary/30 rounded-full"></div>
                  <div className="h-1 flex-1 bg-primary rounded-full"></div>
                </div>
                <div className="flex items-center gap-2 mb-1">
                  <button type="button" onClick={() => setStep(1)} className="text-muted-foreground hover:text-foreground transition-colors p-1 -ml-1">
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <p className="text-xs text-muted-foreground font-display uppercase tracking-wider">Patient Details</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5 group">
                    <Label className="text-xs font-display text-muted-foreground group-focus-within:text-primary">Date of Birth</Label>
                    <div className="relative">
                      <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground group-focus-within:text-primary transition-colors" />
                      <Input type="date" value={dateOfBirth} onChange={(e) => setDateOfBirth(e.target.value)} className="pl-9 bg-secondary/20 border-border/40 focus:bg-background transition-all text-sm" />
                    </div>
                  </div>
                  <div className="space-y-1.5 group">
                    <Label className="text-xs font-display text-muted-foreground group-focus-within:text-primary">Gender</Label>
                    <select value={gender} onChange={(e) => setGender(e.target.value)} className="w-full h-10 rounded-md bg-secondary/20 border border-border/40 px-3 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 transition-all focus:bg-background">
                      <option value="">Select</option>
                      {GENDERS.map((g) => <option key={g} value={g}>{g}</option>)}
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5 group">
                    <Label className="text-xs font-display flex items-center gap-1 text-muted-foreground group-focus-within:text-primary"><Droplets className="w-3 h-3" /> Blood Group</Label>
                    <select value={bloodGroup} onChange={(e) => setBloodGroup(e.target.value)} className="w-full h-10 rounded-md bg-secondary/20 border border-border/40 px-3 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 transition-all focus:bg-background">
                      <option value="">Select</option>
                      {BLOOD_GROUPS.map((bg) => <option key={bg} value={bg}>{bg}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1.5 group">
                    <Label className="text-xs font-display flex items-center gap-1 text-muted-foreground group-focus-within:text-primary"><Phone className="w-3 h-3" /> Phone</Label>
                    <Input type="tel" placeholder="+1 (555)..." value={phone} onChange={(e) => setPhone(e.target.value)} className="bg-secondary/20 border-border/40 focus:bg-background transition-all text-sm" />
                  </div>
                </div>

                <div className="space-y-1.5 group">
                  <Label className="text-xs font-display flex items-center gap-1 text-muted-foreground group-focus-within:text-primary"><AlertTriangle className="w-3 h-3" /> Emergency Contact</Label>
                  <Input placeholder="Name & phone number" value={emergencyContact} onChange={(e) => setEmergencyContact(e.target.value)} className="bg-secondary/20 border-border/40 focus:bg-background transition-all text-sm" />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5 group">
                    <Label className="text-xs font-display flex items-center gap-1 text-muted-foreground group-focus-within:text-primary"><Heart className="w-3 h-3" /> Conditions</Label>
                    <textarea placeholder="e.g. Diabetes..." value={medicalConditions} onChange={(e) => setMedicalConditions(e.target.value)} className="w-full min-h-[50px] rounded-md bg-secondary/20 border border-border/40 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 resize-none focus:bg-background transition-all" />
                  </div>
                  <div className="space-y-1.5 group">
                    <Label className="text-xs font-display flex items-center gap-1 text-muted-foreground group-focus-within:text-primary">Allergies</Label>
                    <textarea placeholder="e.g. Peanuts..." value={allergies} onChange={(e) => setAllergies(e.target.value)} className="w-full min-h-[50px] rounded-md bg-secondary/20 border border-border/40 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 resize-none focus:bg-background transition-all" />
                  </div>
                </div>

                <Button type="submit" className="w-full glow-indigo h-11 text-base" size="lg" disabled={loading}>
                  {loading ? "Creating account..." : "Create Account"} <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
                <p className="text-[10px] text-center text-muted-foreground opacity-70">
                  By clicking Create Account, you agree to our Terms of Service and Privacy Policy.
                </p>
              </form>
            )}
          </div>
        </div>

        <p className="text-center text-xs text-muted-foreground mt-8">
          <Link to="/" className="hover:text-primary transition-colors flex items-center justify-center gap-1">
            <ChevronLeft className="w-3 h-3" /> Back to home
          </Link>
        </p>
      </div>
    </div>
  );
}
