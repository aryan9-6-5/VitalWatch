import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { Activity, Heart, Droplets, Wind, Thermometer, Calendar, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getPatients, getPatientReadings, getPatientTrend, type Patient, type VitalReading, type TrendPoint } from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts";

const familyMembers = [
  { id: "mom", name: "Mom", avatar: "M", color: "from-pink-400 to-pink-600" },
  { id: "dad", name: "Dad", avatar: "D", color: "from-blue-400 to-blue-600" }
];

const vitalTypes = [
  { key: "heartRate", label: "Heart Rate", unit: "BPM", icon: Heart, color: "text-red-500" },
  { key: "bloodPressure", label: "Blood Pressure", unit: "mmHg", icon: Droplets, color: "text-blue-500" },
  { key: "oxygen", label: "Oxygen Level", unit: "%", icon: Wind, color: "text-green-500" },
  { key: "temperature", label: "Temperature", unit: "°F", icon: Thermometer, color: "text-orange-500" }
];

const mockData = {
  mom: {
    heartRate: [
      { time: "00:00", value: 72 },
      { time: "04:00", value: 68 },
      { time: "08:00", value: 78 },
      { time: "12:00", value: 82 },
      { time: "16:00", value: 75 },
      { time: "20:00", value: 73 }
    ],
    bloodPressure: [
      { time: "00:00", systolic: 118, diastolic: 78 },
      { time: "04:00", systolic: 115, diastolic: 75 },
      { time: "08:00", systolic: 122, diastolic: 82 },
      { time: "12:00", systolic: 125, diastolic: 85 },
      { time: "16:00", systolic: 120, diastolic: 80 },
      { time: "20:00", systolic: 119, diastolic: 79 }
    ],
    oxygen: [
      { time: "00:00", value: 99 },
      { time: "04:00", value: 98 },
      { time: "08:00", value: 99 },
      { time: "12:00", value: 97 },
      { time: "16:00", value: 98 },
      { time: "20:00", value: 99 }
    ],
    temperature: [
      { time: "00:00", value: 98.2 },
      { time: "04:00", value: 98.0 },
      { time: "08:00", value: 98.4 },
      { time: "12:00", value: 98.6 },
      { time: "16:00", value: 98.3 },
      { time: "20:00", value: 98.4 }
    ]
  },
  dad: {
    heartRate: [
      { time: "00:00", value: 88 },
      { time: "04:00", value: 85 },
      { time: "08:00", value: 95 },
      { time: "12:00", value: 98 },
      { time: "16:00", value: 92 },
      { time: "20:00", value: 89 }
    ],
    bloodPressure: [
      { time: "00:00", systolic: 138, diastolic: 88 },
      { time: "04:00", systolic: 135, diastolic: 85 },
      { time: "08:00", systolic: 142, diastolic: 92 },
      { time: "12:00", systolic: 145, diastolic: 95 },
      { time: "16:00", systolic: 140, diastolic: 90 },
      { time: "20:00", systolic: 139, diastolic: 89 }
    ],
    oxygen: [
      { time: "00:00", value: 98 },
      { time: "04:00", value: 97 },
      { time: "08:00", value: 98 },
      { time: "12:00", value: 96 },
      { time: "16:00", value: 97 },
      { time: "20:00", value: 98 }
    ],
    temperature: [
      { time: "00:00", value: 98.4 },
      { time: "04:00", value: 98.2 },
      { time: "08:00", value: 98.6 },
      { time: "12:00", value: 99.1 },
      { time: "16:00", value: 98.8 },
      { time: "20:00", value: 98.6 }
    ]
  }
};

const currentVitals = {
  mom: {
    heartRate: { value: 78, status: "normal", trend: "stable" },
    bloodPressure: { value: "120/80", status: "optimal", trend: "improving" },
    oxygen: { value: 99, status: "excellent", trend: "stable" },
    temperature: { value: 98.4, status: "normal", trend: "stable" }
  },
  dad: {
    heartRate: { value: 95, status: "elevated", trend: "increasing" },
    bloodPressure: { value: "140/90", status: "high", trend: "concerning" },
    oxygen: { value: 98, status: "good", trend: "stable" },
    temperature: { value: 98.6, status: "normal", trend: "stable" }
  }
};

export default function Vitals() {
  const [members, setMembers] = useState<Array<{ id: string; name: string; avatar: string; color: string }>>([
    { id: "mom", name: "Mom", avatar: "M", color: "from-pink-400 to-pink-600" },
    { id: "dad", name: "Dad", avatar: "D", color: "from-blue-400 to-blue-600" }
  ]);
  const [selectedMember, setSelectedMember] = useState("mom");
  const [selectedVital, setSelectedVital] = useState("heartRate");
  const [timeRange, setTimeRange] = useState("today");
  const [activeReadings, setActiveReadings] = useState<VitalReading[]>([]);
  const [activeTrend, setActiveTrend] = useState<TrendPoint[]>([]);

  useEffect(() => {
    async function loadPatients() {
      try {
        const pts = await getPatients();
        if (pts.length > 0) {
          const mappedMembers = pts.map((p, i) => ({
            id: p.id,
            name: p.name,
            avatar: p.name[0],
            color: ["from-pink-400 to-pink-600", "from-blue-400 to-blue-600", "from-purple-400 to-purple-600"][i % 3],
          }));
          setMembers(mappedMembers);
          setSelectedMember(pts[0].id);
        }
      } catch (err) {
        console.error("Failed to load patients for vitals:", err);
      }
    }
    loadPatients();
  }, []);

  useEffect(() => {
    if (selectedMember === "mom" || selectedMember === "dad") return;
    
    async function loadVitals() {
      try {
        const [readings, trend] = await Promise.all([
          getPatientReadings(selectedMember),
          getPatientTrend(selectedMember)
        ]);
        setActiveReadings(readings);
        setActiveTrend(trend);
      } catch (err) {
        console.error("Failed to load vitals for patient:", err);
      }
    }
    loadVitals();
  }, [selectedMember]);

  const isMock = selectedMember === "mom" || selectedMember === "dad";

  const getVitalStatus = (vitalKey: string, val: number) => {
    if (vitalKey === "heartRate") {
      if (val > 100) return { value: Math.round(val), status: "elevated", trend: "increasing" };
      if (val < 60) return { value: Math.round(val), status: "low", trend: "decreasing" };
      return { value: Math.round(val), status: "normal", trend: "stable" };
    }
    if (vitalKey === "bloodPressure") {
      return { value: `${Math.round(val)}/${Math.round(val - 40)}`, status: val > 140 ? "high" : "optimal", trend: "stable" };
    }
    if (vitalKey === "oxygen") {
      return { value: Math.round(val), status: val < 95 ? "critical" : "excellent", trend: "stable" };
    }
    if (vitalKey === "temperature") {
      return { value: Number(val.toFixed(1)), status: val > 38 ? "elevated" : "normal", trend: "stable" };
    }
    return { value: Math.round(val), status: "normal", trend: "stable" };
  };

  const currentPatientVitals = !isMock && activeReadings.length > 0
    ? {
        heartRate: getVitalStatus("heartRate", activeReadings[0].raw_inputs?.heart_rate || 75),
        bloodPressure: getVitalStatus("bloodPressure", activeReadings[0].raw_inputs?.systolic_bp || 120),
        oxygen: getVitalStatus("oxygen", activeReadings[0].raw_inputs?.spo2 || 98),
        temperature: getVitalStatus("temperature", activeReadings[0].raw_inputs?.temperature || 98.4),
      }
    : currentVitals[selectedMember as keyof typeof currentVitals] || currentVitals.mom;

  const chartData = !isMock && activeTrend.length > 0
    ? activeTrend.map(t => {
        const time = new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        return {
          time,
          value: selectedVital === "heartRate" ? t.heart_rate
                 : selectedVital === "oxygen" ? t.spo2
                 : selectedVital === "temperature" ? 98.4
                 : t.risk_score * 100,
          systolic: t.systolic_bp || 120,
          diastolic: (t.systolic_bp || 120) - 40,
        };
      })
    : mockData[selectedMember as keyof typeof mockData]?.[selectedVital as keyof typeof mockData.mom] || mockData.mom.heartRate;

  const getStatusColor = (status: string) => {
    switch (status) {
      case "optimal": case "excellent": case "normal": return "text-vitals-healthy";
      case "elevated": case "high": case "concerning": return "text-vitals-warning";
      case "critical": return "text-vitals-critical";
      default: return "text-gray-600";
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving": return <TrendingUp className="w-4 h-4 text-vitals-healthy" />;
      case "increasing": case "concerning": return <TrendingUp className="w-4 h-4 text-vitals-warning" />;
      case "critical": return <AlertTriangle className="w-4 h-4 text-vitals-critical" />;
      default: return <TrendingDown className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-vitals-card-bg to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          className="flex justify-between items-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <h1 className="font-poppins font-bold text-3xl text-gray-800">Real-Time Vitals Monitoring</h1>
            <p className="text-gray-600 mt-2">Track and analyze family health metrics with precision</p>
          </div>
          <div className="flex items-center space-x-4">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="glass-morphism border-0 w-32" data-testid="time-range-select">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="today">Today</SelectItem>
                <SelectItem value="week">This Week</SelectItem>
                <SelectItem value="month">This Month</SelectItem>
              </SelectContent>
            </Select>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button className="bg-vitals-primary text-white hover:bg-blue-600" data-testid="export-data">
                <Calendar className="w-4 h-4 mr-2" />
                Export Data
              </Button>
            </motion.div>
          </div>
        </motion.div>

        {/* Family Member Selector */}
        <motion.div
          className="flex justify-center mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="flex items-center space-x-1 bg-gray-100 rounded-xl p-1">
            {members.map((member) => (
              <motion.button
                key={member.id}
                className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center space-x-3 ${
                  selectedMember === member.id ? "bg-white text-vitals-primary shadow-sm" : "text-gray-600"
                }`}
                onClick={() => setSelectedMember(member.id)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                data-testid={`member-${member.id}`}
              >
                <div className={`w-8 h-8 bg-gradient-to-br ${member.color} rounded-xl flex items-center justify-center`}>
                  <span className="text-white font-semibold text-sm">{member.avatar}</span>
                </div>
                <span>{member.name}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Current Vitals Grid */}
        <motion.div
          className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {vitalTypes.map((vital) => {
            const currentData = currentPatientVitals[vital.key as keyof typeof currentPatientVitals];
            return (
              <motion.div
                key={vital.key}
                className={`glass-morphism-dark rounded-2xl p-6 cursor-pointer transition-all ${
                  selectedVital === vital.key ? "ring-2 ring-vitals-primary" : ""
                }`}
                onClick={() => setSelectedVital(vital.key)}
                whileHover={{ y: -5, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                data-testid={`vital-card-${vital.key}`}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className={`w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center`}>
                    <vital.icon className={`w-6 h-6 ${vital.color}`} />
                  </div>
                  {getTrendIcon(currentData.trend)}
                </div>
                <div className="space-y-2">
                  <h3 className="font-semibold text-gray-800">{vital.label}</h3>
                  <div className="text-2xl font-bold font-mono text-gray-800">
                    {currentData.value}{vital.unit !== "mmHg" && vital.unit}
                  </div>
                  <div className={`text-sm font-medium ${getStatusColor(currentData.status)}`}>
                    {currentData.status.charAt(0).toUpperCase() + currentData.status.slice(1)}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Charts Section */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Chart */}
          <motion.div
            className="lg:col-span-2 glass-morphism-dark rounded-2xl p-6"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-semibold text-lg text-gray-800">
                {vitalTypes.find(v => v.key === selectedVital)?.label} Trends
              </h3>
              <div className="flex items-center space-x-2">
                <Activity className="w-5 h-5 text-vitals-primary" />
                <span className="text-sm text-gray-600">Live monitoring</span>
              </div>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                {selectedVital === "bloodPressure" ? (
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                    <XAxis dataKey="time" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.9)', 
                        border: 'none', 
                        borderRadius: '12px',
                        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)'
                      }} 
                    />
                    <Line 
                      type="monotone" 
                      dataKey="systolic" 
                      stroke="#3b82f6" 
                      strokeWidth={3}
                      dot={{ fill: '#3b82f6', strokeWidth: 2, r: 6 }}
                      name="Systolic"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="diastolic" 
                      stroke="#10b981" 
                      strokeWidth={3}
                      dot={{ fill: '#10b981', strokeWidth: 2, r: 6 }}
                      name="Diastolic"
                    />
                  </LineChart>
                ) : (
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                    <XAxis dataKey="time" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.9)', 
                        border: 'none', 
                        borderRadius: '12px',
                        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)'
                      }} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#3b82f6" 
                      fill="url(#colorGradient)"
                      strokeWidth={3}
                    />
                    <defs>
                      <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Insights Panel */}
          <motion.div
            className="space-y-6"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            {/* Alerts */}
            <div className="glass-morphism-dark rounded-2xl p-6">
              <h3 className="font-semibold text-lg text-gray-800 mb-4 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-vitals-warning" />
                Recent Alerts
              </h3>
              <div className="space-y-3">
                {selectedMember === "dad" ? (
                  <>
                    <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-3">
                      <div className="text-sm font-medium text-vitals-warning">Elevated BP</div>
                      <div className="text-xs text-gray-500">15 minutes ago</div>
                    </div>
                    <div className="bg-red-50 border border-red-200 rounded-xl p-3">
                      <div className="text-sm font-medium text-vitals-critical">High HR Alert</div>
                      <div className="text-xs text-gray-500">1 hour ago</div>
                    </div>
                  </>
                ) : (
                  <div className="bg-green-50 border border-green-200 rounded-xl p-3">
                    <div className="text-sm font-medium text-vitals-healthy">All vitals normal</div>
                    <div className="text-xs text-gray-500">Current status</div>
                  </div>
                )}
              </div>
            </div>

            {/* Device Status */}
            <div className="glass-morphism-dark rounded-2xl p-6">
              <h3 className="font-semibold text-lg text-gray-800 mb-4">Connected Devices</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-white/50 rounded-xl">
                  <div>
                    <div className="text-sm font-medium">Omron BP Monitor</div>
                    <div className="text-xs text-gray-500">Last sync: 2 min ago</div>
                  </div>
                  <div className="w-3 h-3 bg-vitals-healthy rounded-full"></div>
                </div>
                <div className="flex items-center justify-between p-3 bg-white/50 rounded-xl">
                  <div>
                    <div className="text-sm font-medium">Wellue O2Ring</div>
                    <div className="text-xs text-gray-500">Last sync: 5 min ago</div>
                  </div>
                  <div className="w-3 h-3 bg-vitals-healthy rounded-full"></div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className="glass-morphism-dark rounded-2xl p-6">
              <h3 className="font-semibold text-lg text-gray-800 mb-4">Health Recommendations</h3>
              <div className="space-y-3">
                {selectedMember === "dad" ? (
                  <>
                    <div className="text-sm text-gray-700">
                      💧 Reduce sodium intake to help lower blood pressure
                    </div>
                    <div className="text-sm text-gray-700">
                      🚶‍♂️ 30-minute daily walks recommended
                    </div>
                    <div className="text-sm text-gray-700">
                      📅 Schedule cardiology consultation
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-sm text-gray-700">
                      ✅ Maintain current healthy lifestyle
                    </div>
                    <div className="text-sm text-gray-700">
                      💊 Continue Vitamin D supplementation
                    </div>
                    <div className="text-sm text-gray-700">
                      🏃‍♀️ Regular exercise is benefiting your health
                    </div>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
