import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { auth, db } from "@/lib/firebase";
import { collection, onSnapshot, addDoc, deleteDoc, updateDoc, doc, query } from "firebase/firestore";
import { onAuthStateChanged } from "firebase/auth";

export interface FamilyMember {
  id: string;
  name: string;
  relation: string;
  avatar: string;
  accentColor: string;
  isPrimary?: boolean;
  email?: string;
  status?: "active" | "pending";
  linkedUserId?: string;
}

interface FamilyContextType {
  members: FamilyMember[];
  activeMember: FamilyMember | null;
  setActiveMember: (member: FamilyMember) => void;
  addMember: (name: string, relation: string, email?: string) => Promise<void>;
  removeMember: (id: string) => Promise<void>;
  simulateAcceptInvite: (id: string) => Promise<void>;
  loading: boolean;
}

const FamilyContext = createContext<FamilyContextType | undefined>(undefined);

const COLORS = [
  "hsl(270 60% 58%)",
  "hsl(200 80% 55%)",
  "hsl(330 70% 55%)",
  "hsl(150 60% 45%)",
  "hsl(30 80% 55%)",
];

const getInitials = (name: string) => {
  return name && name.length > 0
    ? name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2)
    : "??";
};

export function FamilyProvider({ children }: { children: ReactNode }) {
  const [primaryMember, setPrimaryMember] = useState<FamilyMember | null>(null);
  const [familyMembers, setFamilyMembers] = useState<FamilyMember[]>([]);
  const [activeMember, setActiveMember] = useState<FamilyMember | null>(null);
  const [loading, setLoading] = useState(true);

  // 1. Auth & Primary User Listener
  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, (user) => {
      if (!user) {
        setPrimaryMember(null);
        setFamilyMembers([]);
        setActiveMember(null);
        setLoading(false);
        return;
      }

      const userRef = doc(db, "users", user.uid);
      const unsubscribeUser = onSnapshot(userRef, (docSnap) => {
        const userData = docSnap.data();
        let primaryName = userData?.full_name;

        if (!primaryName) {
          primaryName = user.displayName;
        }
        if (!primaryName && user.email) {
          primaryName = user.email.split("@")[0];
        }
        primaryName = primaryName || "User";

        const newPrimary: FamilyMember = {
          id: "primary",
          name: primaryName,
          relation: "Self",
          avatar: getInitials(primaryName),
          accentColor: "hsl(var(--neurora-indigo))",
          isPrimary: true,
          status: "active",
        };
        setPrimaryMember(newPrimary);

        setActiveMember((prev) => (!prev ? newPrimary : prev));
      }, (error) => {
        console.error("Error fetching user:", error);
      });

      return () => unsubscribeUser();
    });

    return () => unsubscribeAuth();
  }, []);

  // 2. Family Members Listener
  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, (user) => {
      if (!user) return;

      const familyRef = collection(db, "users", user.uid, "family_members");
      const q = query(familyRef);

      const unsubscribeFamily = onSnapshot(q, (snapshot) => {
        const members: FamilyMember[] = snapshot.docs.map((doc, index) => ({
          id: doc.id,
          ...doc.data(),
          status: doc.data().status || "active",
          // ensure legacy members without colors get one
          accentColor: doc.data().accentColor || COLORS[index % COLORS.length],
          avatar: doc.data().avatar || getInitials(doc.data().name || "??"),
        })) as FamilyMember[];

        setFamilyMembers(members);
        setLoading(false);
      }, (error) => {
        console.error("Error fetching family:", error);
        setLoading(false);
      });

      return () => unsubscribeFamily();
    });

    return () => unsubscribeAuth();
  }, []);

  // Combine primary + family
  const allMembers = primaryMember ? [primaryMember, ...familyMembers] : [];

  const addMember = async (name: string, relation: string, email?: string) => {
    if (!auth.currentUser) return;
    try {
      const avatar = getInitials(name);
      const accentColor = COLORS[Math.floor(Math.random() * COLORS.length)];

      await addDoc(collection(db, "users", auth.currentUser.uid, "family_members"), {
        name,
        relation,
        email: email || "",
        status: email ? "pending" : "active", // If email provided, start as pending
        avatar,
        accentColor,
        createdAt: new Date(),
      });
    } catch (error) {
      console.error("Error adding family member:", error);
      throw error;
    }
  };

  const removeMember = async (id: string) => {
    if (!auth.currentUser) return;
    try {
      await deleteDoc(doc(db, "users", auth.currentUser.uid, "family_members", id));
      if (activeMember?.id === id && primaryMember) {
        setActiveMember(primaryMember);
      }
    } catch (error) {
      console.error("Error removing family member:", error);
      throw error;
    }
  };

  const simulateAcceptInvite = async (id: string) => {
    if (!auth.currentUser) return;
    try {
      // In a real app, this would be triggered by the invited user accepting an email link.
      // Here, we just update the status to 'active' and maybe assign a mock linkedUserId
      await updateDoc(doc(db, "users", auth.currentUser.uid, "family_members", id), {
        status: "active",
        linkedUserId: "mock_linked_user_" + id // for now, just a placeholder
      });
    } catch (error) {
      console.error("Error accepting invite:", error);
      throw error;
    }
  };

  return (
    <FamilyContext.Provider value={{
      members: allMembers,
      activeMember: activeMember || primaryMember,
      setActiveMember,
      addMember,
      removeMember,
      simulateAcceptInvite,
      loading
    }}>
      {children}
    </FamilyContext.Provider>
  );
}

export function useFamily() {
  const ctx = useContext(FamilyContext);
  if (!ctx) throw new Error("useFamily must be used within FamilyProvider");
  return ctx;
}
