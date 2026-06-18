import { motion } from "framer-motion";
import { Pill, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Patient {
  id: string;
  name: string;
  age: number;
  initials: string;
  status: "critical" | "stable" | "warning";
  vitals: {
    bp: string;
    hr: number;
    spo2: number;
    temp: number;
  };
  lastAlert: string;
  alertTime: string;
  bgColor: string;
}

interface PatientCardProps {
  patient: Patient;
}

export default function PatientCard({ patient }: PatientCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "critical": return "border-vitals-critical";
      case "stable": return "border-vitals-healthy";
      case "warning": return "border-vitals-warning";
      default: return "border-gray-200";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "critical": return { color: "text-vitals-critical", label: "CRITICAL", bg: "bg-vitals-critical" };
      case "stable": return { color: "text-vitals-healthy", label: "STABLE", bg: "bg-vitals-healthy" };
      case "warning": return { color: "text-vitals-warning", label: "MONITOR", bg: "bg-vitals-warning" };
      default: return { color: "text-gray-500", label: "UNKNOWN", bg: "bg-gray-500" };
    }
  };

  const getVitalColor = (status: string, vital: string) => {
    if (vital === "bp" && status === "critical") return "text-vitals-critical";
    if (vital === "hr" && status === "warning") return "text-vitals-warning";
    if (vital === "temp" && status === "warning") return "text-vitals-warning";
    return "text-vitals-healthy";
  };

  const statusBadge = getStatusBadge(patient.status);

  return (
    <motion.div
      className={`glass-morphism-dark rounded-2xl p-6 border-l-4 ${getStatusColor(patient.status)} hover:shadow-xl transition-all cursor-pointer`}
      whileHover={{ y: -5, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      data-testid={`patient-card-${patient.id}`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-12 h-12 bg-gradient-to-br ${patient.bgColor} rounded-xl flex items-center justify-center`}>
            <span className="text-white font-semibold text-lg">{patient.initials}</span>
          </div>
          <div>
            <h3 className="font-semibold text-gray-800">{patient.name}</h3>
            <p className="text-sm text-gray-500">Age {patient.age} • ID: {patient.id}</p>
          </div>
        </div>
        <div className="flex items-center space-x-1">
          {patient.status === "critical" || patient.status === "warning" ? (
            <motion.div
              className={`w-3 h-3 ${statusBadge.bg} rounded-full`}
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          ) : (
            <div className={`w-3 h-3 ${statusBadge.bg} rounded-full`} />
          )}
          <span className={`text-xs ${statusBadge.color} font-medium`}>{statusBadge.label}</span>
        </div>
      </div>

      {/* Vitals Display */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="text-center bg-white/50 rounded-xl p-3">
          <div className={`font-mono text-lg font-bold ${getVitalColor(patient.status, "bp")}`}>
            {patient.vitals.bp}
          </div>
          <div className="text-xs text-gray-500">BP (mmHg)</div>
        </div>
        <div className="text-center bg-white/50 rounded-xl p-3">
          <div className={`font-mono text-lg font-bold ${getVitalColor(patient.status, "hr")}`}>
            {patient.vitals.hr}
          </div>
          <div className="text-xs text-gray-500">HR (BPM)</div>
        </div>
        <div className="text-center bg-white/50 rounded-xl p-3">
          <div className="font-mono text-lg font-bold text-vitals-healthy">
            {patient.vitals.spo2}
          </div>
          <div className="text-xs text-gray-500">SpO₂ (%)</div>
        </div>
        <div className="text-center bg-white/50 rounded-xl p-3">
          <div className={`font-mono text-lg font-bold ${getVitalColor(patient.status, "temp")}`}>
            {patient.vitals.temp}
          </div>
          <div className="text-xs text-gray-500">Temp (°F)</div>
        </div>
      </div>

      {/* Last Alert */}
      <div className={`${patient.status === "critical" ? "bg-red-50 border-red-200" : 
                       patient.status === "warning" ? "bg-yellow-50 border-yellow-200" : 
                       "bg-green-50 border-green-200"} border rounded-xl p-3 mb-4`}>
        <div className="flex items-center justify-between">
          <div className={`text-sm font-medium ${statusBadge.color}`}>{patient.lastAlert}</div>
          <div className="text-xs text-gray-500">{patient.alertTime}</div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-2">
        <Button 
          variant="default" 
          size="sm" 
          className="flex-1 bg-vitals-primary hover:bg-blue-600"
          data-testid={`view-details-${patient.id}`}
        >
          View Details
        </Button>
        <Button 
          variant="outline" 
          size="sm" 
          className="bg-green-500 text-white hover:bg-green-600 border-green-500"
          data-testid={`prescribe-${patient.id}`}
        >
          <Pill className="w-4 h-4" />
        </Button>
        <Button 
          variant="outline" 
          size="sm" 
          className="bg-blue-500 text-white hover:bg-blue-600 border-blue-500"
          data-testid={`message-${patient.id}`}
        >
          <MessageCircle className="w-4 h-4" />
        </Button>
      </div>
    </motion.div>
  );
}
