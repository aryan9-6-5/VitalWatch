import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Link } from "wouter";

interface Vital {
  value: string | number;
  emoji: string;
  status: string;
  color: string;
}

interface FamilyMember {
  id: string;
  name: string;
  fullName: string;
  age: number;
  status: "healthy" | "warning" | "critical";
  avatar: string;
  vitals: {
    heartRate: Vital;
    bloodPressure: Vital;
    oxygen: Vital;
    temperature: Vital;
  };
}

interface VitalsCardProps {
  member: FamilyMember;
}

export default function VitalsCard({ member }: VitalsCardProps) {
  const getStatusIndicator = () => {
    switch (member.status) {
      case "healthy":
        return { color: "bg-vitals-healthy", text: "All Good", textColor: "text-vitals-healthy" };
      case "warning":
        return { color: "bg-vitals-warning", text: "Needs Attention", textColor: "text-vitals-warning" };
      case "critical":
        return { color: "bg-vitals-critical", text: "Critical", textColor: "text-vitals-critical" };
    }
  };

  const statusIndicator = getStatusIndicator();
  const borderClass = member.status === "warning" ? "border-l-4 border-vitals-warning" : "";

  return (
    <motion.div
      className={`glass-morphism-dark rounded-3xl p-8 hover:shadow-2xl transition-all ${
        member.status === "healthy" ? "vitals-glow" : ""
      } ${borderClass}`}
      whileHover={{ y: -5, scale: 1.02 }}
      data-testid={`vitals-card-${member.id}`}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <motion.img
            src={member.avatar}
            alt={member.name}
            className="w-16 h-16 rounded-2xl object-cover"
            whileHover={{ scale: 1.1 }}
          />
          <div>
            <h3 className="font-poppins font-semibold text-xl text-gray-800">{member.name}</h3>
            <p className="text-gray-500">{member.fullName} • Age {member.age}</p>
            <div className="flex items-center space-x-2 mt-1">
              <motion.div
                className={`w-3 h-3 ${statusIndicator.color} rounded-full`}
                animate={member.status !== "healthy" ? { scale: [1, 1.2, 1] } : {}}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className={`text-sm ${statusIndicator.textColor} font-medium`}>
                {statusIndicator.text}
              </span>
            </div>
          </div>
        </div>
        {member.status === "warning" && (
          <div className="bg-yellow-100 text-vitals-warning px-3 py-1 rounded-full text-sm font-medium">
            Monitor BP
          </div>
        )}
      </div>

      {/* Vitals with Emojis */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {Object.entries(member.vitals).map(([key, vital]) => (
          <motion.div
            key={key}
            className={`glass-morphism rounded-2xl p-4 text-center hover:bg-blue-50 transition-colors ${
              key === "bloodPressure" && member.status === "warning" ? "border border-yellow-200" : ""
            }`}
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl mb-2">{vital.emoji}</div>
            <div className={`font-mono text-xl font-bold ${vital.color}`}>{vital.value}</div>
            <div className="text-sm text-gray-600">
              {key === "heartRate" && "Heart Rate"}
              {key === "bloodPressure" && "Blood Pressure"}
              {key === "oxygen" && "Oxygen"}
              {key === "temperature" && "Temperature"}
            </div>
            <div className="text-xs text-green-600">{vital.status}</div>
          </motion.div>
        ))}
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-3">
        <Link href="/ai-assistant" data-testid={`ask-doctor-${member.id}`} className="flex-1">
          <Button className="w-full bg-vitals-primary text-white hover:bg-blue-600 transition-colors font-medium px-4 py-3 rounded-xl">
            💬 Ask Doctor
          </Button>
        </Link>
        <Link href="/vitals" data-testid={`view-history-${member.id}`} className="flex-1">
          <Button variant="outline" className="w-full glass-morphism text-vitals-primary hover:bg-blue-50 transition-colors font-medium px-4 py-3 rounded-xl border-0">
            📊 View History
          </Button>
        </Link>
      </div>
    </motion.div>
  );
}
