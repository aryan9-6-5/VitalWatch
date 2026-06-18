import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { useParams } from "wouter";
import { 
  Activity, Heart, Droplets, Wind, Thermometer, Calendar, TrendingUp, TrendingDown, 
  AlertTriangle, Shield, Brain, Pill, Clock, Target, BarChart3, PieChart, Zap,
  User, Phone, MapPin, FileText, Settings, Download, Share2, Bell
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { getPatient, getPatientReadings, getPatientAlerts, getPatientTrend, type Patient, type VitalReading, type TrendPoint } from "@/lib/api";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, BarChart, Bar, PieChart as RechartsPieChart, Cell, Pie,
  RadialBarChart, RadialBar, ComposedChart
} from "recharts";

interface PatientData {
  id: string;
  name: string;
  fullName: string;
  age: number;
  gender: string;
  avatar: string;
  contactInfo: {
    phone: string;
    emergency: string;
    address: string;
  };
  conditions: string[];
  medications: Array<{
    name: string;
    dosage: string;
    frequency: string;
    adherence: number;
  }>;
  riskScore: {
    overall: number;
    cardiovascular: number;
    diabetes: number;
    hypertension: number;
  };
  healthScore: number;
  lastVisit: string;
  nextAppointment: string;
}

const mockPatientData: PatientData = {
  id: "P002",
  name: "Priya Sharma",
  fullName: "Priya Sharma",
  age: 58,
  gender: "Female",
  avatar: "https://images.unsplash.com/photo-1582750433449-648ed127bb54?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&h=150",
  contactInfo: {
    phone: "+91 98765 43210",
    emergency: "+91 98765 43211 (Daughter)",
    address: "123 MG Road, Bangalore, Karnataka"
  },
  conditions: ["Hypertension", "Pre-diabetes", "Anxiety"],
  medications: [
    { name: "Amlodipine", dosage: "5mg", frequency: "Once daily", adherence: 92 },
    { name: "Metformin", dosage: "500mg", frequency: "Twice daily", adherence: 88 },
    { name: "Vitamin D3", dosage: "2000 IU", frequency: "Once daily", adherence: 75 }
  ],
  riskScore: {
    overall: 35,
    cardiovascular: 28,
    diabetes: 42,
    hypertension: 31
  },
  healthScore: 78,
  lastVisit: "2024-01-15",
  nextAppointment: "2024-02-15"
};

const vitalTrends = {
  heartRate: [
    { time: "Mon", morning: 72, afternoon: 78, evening: 75, target: 75 },
    { time: "Tue", morning: 74, afternoon: 82, evening: 77, target: 75 },
    { time: "Wed", morning: 71, afternoon: 79, evening: 76, target: 75 },
    { time: "Thu", morning: 73, afternoon: 81, evening: 78, target: 75 },
    { time: "Fri", morning: 72, afternoon: 85, evening: 79, target: 75 },
    { time: "Sat", morning: 70, afternoon: 77, evening: 74, target: 75 },
    { time: "Sun", morning: 69, afternoon: 76, evening: 73, target: 75 }
  ],
  bloodPressure: [
    { time: "Week 1", systolic: 135, diastolic: 85, target: 120 },
    { time: "Week 2", systolic: 132, diastolic: 82, target: 120 },
    { time: "Week 3", systolic: 128, diastolic: 80, target: 120 },
    { time: "Week 4", systolic: 125, diastolic: 78, target: 120 }
  ]
};

const medicationAdherence = [
  { name: "Amlodipine", adherence: 92, target: 95, color: "#10b981" },
  { name: "Metformin", adherence: 88, target: 95, color: "#3b82f6" },
  { name: "Vitamin D3", adherence: 75, target: 95, color: "#f59e0b" }
];

const riskDistribution = [
  { name: "Cardiovascular", value: 28, color: "#ef4444" },
  { name: "Diabetes", value: 42, color: "#f59e0b" },
  { name: "Hypertension", value: 31, color: "#8b5cf6" },
  { name: "Other", value: 15, color: "#10b981" }
];

const healthMetrics = [
  { metric: "Physical Activity", score: 85, trend: 5, color: "#10b981" },
  { metric: "Sleep Quality", score: 72, trend: -3, color: "#3b82f6" },
  { metric: "Stress Level", score: 68, trend: 8, color: "#f59e0b" },
  { metric: "Nutrition", score: 91, trend: 12, color: "#10b981" }
];

export default function PatientAnalysis() {
  const [selectedTimeframe, setSelectedTimeframe] = useState("week");
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedMetric, setSelectedMetric] = useState("heartRate");
  
  const { id } = useParams();
  const [patient, setPatient] = useState<PatientData>(mockPatientData);
  const [trends, setTrends] = useState<TrendPoint[]>([]);
  const [readings, setReadings] = useState<VitalReading[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!id) return;
    async function loadPatient() {
      setLoading(true);
      try {
        const [pDetails, pTrend, pReadings] = await Promise.all([
          getPatient(id as string),
          getPatientTrend(id as string).catch(() => []),
          getPatientReadings(id as string).catch(() => []),
        ]);

        const latestReading = pReadings[0];
        const riskVal = latestReading?.risk_score !== undefined ? Math.round(latestReading.risk_score * 100) : 20;

        const mappedPatient: PatientData = {
          id: pDetails.id,
          name: pDetails.name,
          fullName: pDetails.name,
          age: pDetails.age,
          gender: "Unknown",
          avatar: "https://images.unsplash.com/photo-1582750433449-648ed127bb54?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&h=150",
          contactInfo: {
            phone: "+91 98765 43210",
            emergency: "+91 98765 43211",
            address: pDetails.address || "N/A",
          },
          conditions: [pDetails.condition],
          medications: [
            { name: "Amlodipine", dosage: "5mg", frequency: "Once daily", adherence: 92 }
          ],
          riskScore: {
            overall: riskVal,
            cardiovascular: Math.round(riskVal * 0.8),
            diabetes: Math.round(riskVal * 1.2),
            hypertension: Math.round(riskVal * 0.9),
          },
          healthScore: Math.max(0, 100 - riskVal),
          lastVisit: "N/A",
          nextAppointment: "N/A",
        };
        setPatient(mappedPatient);
        setTrends(pTrend);
        setReadings(pReadings);
      } catch (err) {
        console.error("Failed to load patient detail from backend:", err);
      } finally {
        setLoading(false);
      }
    }
    loadPatient();
  }, [id]);

  const patientData = patient;

  const displayTrends = trends.length > 0 
    ? trends.map(t => ({
        time: new Date(t.timestamp).toLocaleDateString(undefined, { weekday: 'short', hour: '2-digit' }),
        value: t.heart_rate || 75,
        morning: (t.heart_rate || 75) - 5,
        afternoon: t.heart_rate || 75,
        evening: (t.heart_rate || 75) + 3,
        target: 75,
        systolic: t.systolic_bp || 120,
        diastolic: (t.systolic_bp || 120) - 40,
      }))
    : vitalTrends.heartRate;

  const displayBPTrends = trends.length > 0
    ? trends.map(t => ({
        time: new Date(t.timestamp).toLocaleDateString(undefined, { weekday: 'short', hour: '2-digit' }),
        systolic: t.systolic_bp || 120,
        diastolic: (t.systolic_bp || 120) - 40,
        target: 120,
      }))
    : vitalTrends.bloodPressure;


  const getRiskColor = (score: number) => {
    if (score >= 70) return "text-red-500";
    if (score >= 40) return "text-yellow-500";
    return "text-green-500";
  };

  const getRiskLevel = (score: number) => {
    if (score >= 70) return "High Risk";
    if (score >= 40) return "Moderate Risk";
    return "Low Risk";
  };

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Enhanced Header with Patient Info */}
        <motion.div
          className="glass-morphism-dark rounded-3xl p-8 mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex justify-between items-start">
            <div className="flex items-start space-x-6">
              <motion.img
                src={patientData.avatar}
                alt={patientData.name}
                className="w-24 h-24 rounded-2xl object-cover shadow-lg"
                whileHover={{ scale: 1.05 }}
              />
              <div className="space-y-3">
                <div>
                  <h1 className="font-poppins font-bold text-3xl text-gray-800">{patientData.fullName}</h1>
                  <p className="text-gray-600 text-lg">Patient ID: {patientData.id} • {patientData.gender}, Age {patientData.age}</p>
                </div>
                
                <div className="flex items-center space-x-6 text-sm text-gray-600">
                  <div className="flex items-center space-x-1">
                    <Phone className="w-4 h-4" />
                    <span>{patientData.contactInfo.phone}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <MapPin className="w-4 h-4" />
                    <span>Bangalore, Karnataka</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Calendar className="w-4 h-4" />
                    <span>Next Visit: {new Date(patientData.nextAppointment).toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  {patientData.conditions.map((condition, index) => (
                    <span key={index} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                      {condition}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <Button variant="outline" className="glass-morphism hover:bg-blue-50">
                <Share2 className="w-4 h-4 mr-2" />
                Share Report
              </Button>
              <Button className="bg-vitals-primary text-white hover:bg-blue-600">
                <Bell className="w-4 h-4 mr-2" />
                Set Alert
              </Button>
              <Button className="bg-green-500 text-white hover:bg-green-600">
                <Phone className="w-4 h-4 mr-2" />
                Contact
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Health Score & Risk Assessment Dashboard */}
        <motion.div
          className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {/* Overall Health Score */}
          <div className="glass-morphism-dark rounded-2xl p-6 text-center">
            <div className="flex items-center justify-center mb-4">
              <div className="relative w-20 h-20">
                <svg className="w-20 h-20 transform -rotate-90">
                  <circle cx="40" cy="40" r="36" stroke="#e5e7eb" strokeWidth="8" fill="transparent" />
                  <circle 
                    cx="40" cy="40" r="36" 
                    stroke="#10b981" 
                    strokeWidth="8" 
                    fill="transparent"
                    strokeLinecap="round"
                    strokeDasharray={`${(patientData.healthScore / 100) * 226.19} 226.19`}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-gray-800">{patientData.healthScore}</span>
                </div>
              </div>
            </div>
            <h3 className="font-semibold text-gray-800 mb-2">Overall Health Score</h3>
            <p className="text-sm text-green-600 font-medium">Good Health</p>
            <div className="flex items-center justify-center mt-2">
              <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
              <span className="text-xs text-green-600">+5 from last month</span>
            </div>
          </div>

          {Object.entries(patientData.riskScore).slice(1).map(([type, score]) => {
            const scoreNum = score as number;
            return (
              <div key={type} className="glass-morphism-dark rounded-2xl p-6">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-gray-800 capitalize">{type} Risk</h3>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    scoreNum >= 70 ? 'bg-red-100 text-red-700' :
                    scoreNum >= 40 ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {getRiskLevel(scoreNum)}
                  </div>
                </div>
                <div className="relative">
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className={`h-3 rounded-full ${
                        scoreNum >= 70 ? 'bg-red-500' :
                        scoreNum >= 40 ? 'bg-yellow-500' :
                        'bg-green-500'
                      }`}
                      style={{ width: `${scoreNum}%` }}
                    ></div>
                  </div>
                  <span className={`text-lg font-bold mt-2 block ${getRiskColor(scoreNum)}`}>
                    {scoreNum}%
                  </span>
                </div>
              </div>
            );
          })}
        </motion.div>

        {/* Navigation Tabs */}
        <motion.div
          className="flex items-center space-x-1 bg-white rounded-2xl p-2 mb-8 shadow-sm"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'vitals', label: 'Vital Trends', icon: Heart },
            { id: 'medications', label: 'Medications', icon: Pill },
            { id: 'lifestyle', label: 'Lifestyle', icon: Target },
            { id: 'reports', label: 'Reports', icon: FileText }
          ].map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all ${
                  activeTab === tab.id 
                    ? 'bg-vitals-primary text-white shadow-lg transform scale-105' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </motion.div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Main Analytics */}
            <div className="lg:col-span-2 space-y-6">
              
              {/* Vital Signs Trends */}
              <motion.div
                className="glass-morphism-dark rounded-2xl p-6"
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="font-semibold text-xl text-gray-800">Comprehensive Vital Analysis</h3>
                  <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                    <SelectTrigger className="glass-morphism border-0 w-40">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="heartRate">Heart Rate</SelectItem>
                      <SelectItem value="bloodPressure">Blood Pressure</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    {selectedMetric === 'heartRate' ? (
                      <ComposedChart data={displayTrends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                        <XAxis dataKey="time" stroke="#6b7280" />
                        <YAxis stroke="#6b7280" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                            border: 'none', 
                            borderRadius: '16px',
                            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)'
                          }} 
                        />
                        <Area 
                          type="monotone" 
                          dataKey="morning" 
                          stroke="#22c55e" 
                          fill="rgba(34, 197, 94, 0.1)"
                          strokeWidth={2}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="afternoon" 
                          stroke="#3b82f6" 
                          strokeWidth={3}
                          dot={{ fill: '#3b82f6', strokeWidth: 2, r: 5 }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="evening" 
                          stroke="#8b5cf6" 
                          strokeWidth={3}
                          dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 5 }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="target" 
                          stroke="#ef4444" 
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                      </ComposedChart>
                    ) : (
                      <ComposedChart data={displayBPTrends}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                        <XAxis dataKey="time" stroke="#6b7280" />
                        <YAxis stroke="#6b7280" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                            border: 'none', 
                            borderRadius: '16px',
                            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)'
                          }} 
                        />
                        <Bar dataKey="systolic" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        <Line 
                          type="monotone" 
                          dataKey="diastolic" 
                          stroke="#10b981" 
                          strokeWidth={3}
                          dot={{ fill: '#10b981', strokeWidth: 2, r: 6 }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="target" 
                          stroke="#ef4444" 
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                        />
                      </ComposedChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </motion.div>

              {/* Medication Adherence */}
              <motion.div
                className="glass-morphism-dark rounded-2xl p-6"
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.8 }}
              >
                <h3 className="font-semibold text-xl text-gray-800 mb-6">Medication Adherence Analysis</h3>
                
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={medicationAdherence} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                      <XAxis type="number" domain={[0, 100]} stroke="#6b7280" />
                      <YAxis type="category" dataKey="name" stroke="#6b7280" width={80} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                          border: 'none', 
                          borderRadius: '16px',
                          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)'
                        }} 
                      />
                      <Bar dataKey="adherence" fill="#3b82f6" radius={[0, 8, 8, 0]} />
                      <Bar dataKey="target" fill="#e5e7eb" radius={[0, 8, 8, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            </div>

            {/* Side Panel */}
            <div className="space-y-6">
              
              {/* Risk Distribution */}
              <motion.div
                className="glass-morphism-dark rounded-2xl p-6"
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <h3 className="font-semibold text-lg text-gray-800 mb-6">Risk Distribution</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Pie
                        data={riskDistribution}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={70}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {riskDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  {riskDistribution.map((item, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                      <span className="text-xs text-gray-600">{item.name}</span>
                    </div>
                  ))}
                </div>
              </motion.div>

              {/* Health Metrics */}
              <motion.div
                className="glass-morphism-dark rounded-2xl p-6"
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.8 }}
              >
                <h3 className="font-semibold text-lg text-gray-800 mb-6">Lifestyle Metrics</h3>
                <div className="space-y-4">
                  {healthMetrics.map((metric, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">{metric.metric}</span>
                        <div className="flex items-center space-x-1">
                          <span className="text-sm font-bold" style={{ color: metric.color }}>
                            {metric.score}%
                          </span>
                          {metric.trend > 0 ? (
                            <TrendingUp className="w-3 h-3 text-green-500" />
                          ) : (
                            <TrendingDown className="w-3 h-3 text-red-500" />
                          )}
                        </div>
                      </div>
                      <div className="relative">
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="h-2 rounded-full transition-all duration-500"
                            style={{ 
                              width: `${metric.score}%`,
                              backgroundColor: metric.color 
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>

              {/* Recent Activities */}
              <motion.div
                className="glass-morphism-dark rounded-2xl p-6"
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 1.0 }}
              >
                <h3 className="font-semibold text-lg text-gray-800 mb-4">Recent Activities</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-xl">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-800">Medication taken on time</p>
                      <p className="text-xs text-gray-500">2 hours ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-xl">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-800">Blood pressure recorded</p>
                      <p className="text-xs text-gray-500">4 hours ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-xl">
                    <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-800">30-min walk completed</p>
                      <p className="text-xs text-gray-500">6 hours ago</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        )}

        {/* Additional tab content would go here */}
        {activeTab !== 'overview' && (
          <motion.div
            className="glass-morphism-dark rounded-2xl p-8 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
          >
            <div className="text-gray-500 mb-4">
              <Settings className="w-16 h-16 mx-auto mb-4 opacity-50" />
            </div>
            <h3 className="font-semibold text-xl text-gray-800 mb-2">
              {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Analysis
            </h3>
            <p className="text-gray-600">
              Detailed {activeTab} analytics and insights coming soon. This will include comprehensive data visualization and predictive analytics.
            </p>
          </motion.div>
        )}
        
      </div>
    </div>
  );
}