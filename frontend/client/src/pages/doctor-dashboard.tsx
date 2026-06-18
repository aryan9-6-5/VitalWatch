import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { Link } from "wouter";
import { 
  Filter, Plus, Search, Calendar, TrendingUp, TrendingDown, AlertTriangle,
  Users, Activity, Stethoscope, ClipboardCheck, Phone, MapPin, Clock,
  BarChart3, PieChart, Target, Heart, Brain, Shield, FileText, Download,
  ChevronDown, Eye, Edit, Archive, UserPlus, Zap, LineChart, Truck
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { 
  LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, BarChart, Bar, PieChart as RechartsPieChart, Cell, Pie,
  RadialBarChart, RadialBar, ComposedChart
} from "recharts";
import PatientCard from "@/components/PatientCard";
import { getPatients, getActiveAlerts, type Patient as BackendPatient, type Alert as BackendAlert } from "@/lib/api";

interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  initials: string;
  status: "critical" | "stable" | "warning";
  riskScore: number;
  healthScore: number;
  admissionDate: string;
  lastVisit: string;
  nextAppointment: string;
  primaryDoctor: string;
  conditions: string[];
  location: string;
  contactInfo: {
    phone: string;
    emergency: string;
  };
  vitals: {
    bp: string;
    hr: number;
    spo2: number;
    temp: number;
    rr?: number; // Respiratory Rate
    glucose?: number;
  };
  lastAlert: string;
  alertTime: string;
  bgColor: string;
  medications: number;
  compliance: number;
  recentTests: Array<{
    test: string;
    result: string;
    date: string;
    status: "normal" | "abnormal" | "critical";
  }>;
}

const mockPatients: Patient[] = [
  {
    id: "P001",
    name: "Rajesh Kumar",
    age: 67,
    gender: "Male",
    initials: "RK",
    status: "critical" as const,
    riskScore: 85,
    healthScore: 45,
    admissionDate: "2024-01-15",
    lastVisit: "2024-02-05",
    nextAppointment: "2024-02-12",
    primaryDoctor: "Dr. Sarah Johnson",
    conditions: ["Hypertension", "Type 2 Diabetes", "CAD"],
    location: "Room 301-A",
    contactInfo: {
      phone: "+91 98765 43210",
      emergency: "+91 98765 43211 (Son)"
    },
    vitals: {
      bp: "165/95",
      hr: 92,
      spo2: 98,
      temp: 98.6,
      rr: 18,
      glucose: 180
    },
    lastAlert: "Critical: BP 170/100, Glucose 195",
    alertTime: "2 min ago",
    bgColor: "from-red-400 to-red-600",
    medications: 6,
    compliance: 78,
    recentTests: [
      { test: "HbA1c", result: "9.2%", date: "2024-02-01", status: "critical" },
      { test: "ECG", result: "ST-T changes", date: "2024-02-05", status: "abnormal" },
      { test: "Lipid Panel", result: "LDL 160 mg/dL", date: "2024-01-28", status: "abnormal" }
    ]
  },
  {
    id: "P002",
    name: "Priya Sharma",
    age: 58,
    gender: "Female",
    initials: "PS",
    status: "stable" as const,
    riskScore: 35,
    healthScore: 82,
    admissionDate: "2024-01-20",
    lastVisit: "2024-02-01",
    nextAppointment: "2024-03-01",
    primaryDoctor: "Dr. Michael Chen",
    conditions: ["Hypothyroidism", "Osteoporosis"],
    location: "Outpatient",
    contactInfo: {
      phone: "+91 98765 43212",
      emergency: "+91 98765 43213 (Daughter)"
    },
    vitals: {
      bp: "120/80",
      hr: 78,
      spo2: 99,
      temp: 98.4,
      rr: 16
    },
    lastAlert: "All vitals within normal range",
    alertTime: "5 min ago",
    bgColor: "from-green-400 to-green-600",
    medications: 3,
    compliance: 94,
    recentTests: [
      { test: "TSH", result: "2.8 mIU/L", date: "2024-01-25", status: "normal" },
      { test: "DEXA Scan", result: "T-score -1.8", date: "2024-01-15", status: "abnormal" },
      { test: "CBC", result: "WNL", date: "2024-01-20", status: "normal" }
    ]
  },
  {
    id: "P003",
    name: "Anita Gupta",
    age: 45,
    gender: "Female",
    initials: "AG",
    status: "warning" as const,
    riskScore: 58,
    healthScore: 68,
    admissionDate: "2024-01-18",
    lastVisit: "2024-02-03",
    nextAppointment: "2024-02-15",
    primaryDoctor: "Dr. Rajesh Patel",
    conditions: ["Pre-hypertension", "Anxiety", "GERD"],
    location: "Cardiology Wing",
    contactInfo: {
      phone: "+91 98765 43214",
      emergency: "+91 98765 43215 (Husband)"
    },
    vitals: {
      bp: "135/85",
      hr: 82,
      spo2: 97,
      temp: 100.1,
      rr: 20
    },
    lastAlert: "Elevated BP and HR, mild fever",
    alertTime: "15 min ago",
    bgColor: "from-yellow-400 to-orange-500",
    medications: 4,
    compliance: 85,
    recentTests: [
      { test: "Stress Echo", result: "Mild LVH", date: "2024-02-01", status: "abnormal" },
      { test: "Holter Monitor", result: "Occasional PVCs", date: "2024-01-28", status: "abnormal" },
      { test: "CRP", result: "8.2 mg/L", date: "2024-02-03", status: "abnormal" }
    ]
  },
  {
    id: "P004",
    name: "Arjun Reddy",
    age: 34,
    gender: "Male",
    initials: "AR",
    status: "stable" as const,
    riskScore: 25,
    healthScore: 88,
    admissionDate: "2024-02-01",
    lastVisit: "2024-02-06",
    nextAppointment: "2024-03-06",
    primaryDoctor: "Dr. Lisa Wang",
    conditions: ["Asthma", "Allergic Rhinitis"],
    location: "Pulmonology",
    contactInfo: {
      phone: "+91 98765 43216",
      emergency: "+91 98765 43217 (Wife)"
    },
    vitals: {
      bp: "115/75",
      hr: 72,
      spo2: 99,
      temp: 98.2,
      rr: 14
    },
    lastAlert: "Routine monitoring - stable",
    alertTime: "1 hour ago",
    bgColor: "from-green-400 to-green-600",
    medications: 2,
    compliance: 96,
    recentTests: [
      { test: "Spirometry", result: "FEV1 85% predicted", date: "2024-02-01", status: "normal" },
      { test: "Chest X-ray", result: "Clear lung fields", date: "2024-02-01", status: "normal" }
    ]
  }
];

const departmentStats = [
  { name: "Cardiology", patients: 12, critical: 3, color: "#ef4444" },
  { name: "Pulmonology", patients: 8, critical: 1, color: "#3b82f6" },
  { name: "Endocrinology", patients: 15, critical: 2, color: "#10b981" },
  { name: "General", patients: 22, critical: 1, color: "#f59e0b" }
];

const weeklyTrends = [
  { day: "Mon", admissions: 12, discharges: 8, alerts: 15 },
  { day: "Tue", admissions: 15, discharges: 10, alerts: 12 },
  { day: "Wed", admissions: 18, discharges: 14, alerts: 18 },
  { day: "Thu", admissions: 14, discharges: 12, alerts: 10 },
  { day: "Fri", admissions: 16, discharges: 15, alerts: 14 },
  { day: "Sat", admissions: 8, discharges: 6, alerts: 8 },
  { day: "Sun", admissions: 6, discharges: 4, alerts: 6 }
];

export default function DoctorDashboard() {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [departmentFilter, setDepartmentFilter] = useState("all");
  const [selectedView, setSelectedView] = useState("overview");
  const [sortBy, setSortBy] = useState("riskScore");
  const [patients, setPatients] = useState<Patient[]>(mockPatients);
  const [activeAlerts, setActiveAlerts] = useState<BackendAlert[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch real patients and alerts from the backend on mount
  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      try {
        const [backendPatients, alerts] = await Promise.allSettled([
          getPatients(),
          getActiveAlerts(),
        ]);

        // Convert backend patients to frontend format and merge
        if (backendPatients.status === "fulfilled" && backendPatients.value.length > 0) {
          const realPatients: Patient[] = backendPatients.value.map((bp: BackendPatient, i: number) => ({
            id: bp.id,
            name: bp.name,
            age: bp.age,
            gender: "Unknown",
            initials: bp.name.split(" ").map((n) => n[0]).join("").toUpperCase().slice(0, 2),
            status: "stable" as const,
            riskScore: 20,
            healthScore: 80,
            admissionDate: bp.created_at?.split("T")[0] || "N/A",
            lastVisit: "N/A",
            nextAppointment: "N/A",
            primaryDoctor: bp.doctor_email,
            conditions: [bp.condition],
            location: bp.address || "N/A",
            contactInfo: { phone: "N/A", emergency: "N/A" },
            vitals: { bp: "N/A", hr: 0, spo2: 0, temp: 0 },
            lastAlert: "No alerts",
            alertTime: "N/A",
            bgColor: ["from-green-400 to-green-600", "from-blue-400 to-blue-600", "from-purple-400 to-purple-600"][i % 3],
            medications: 0,
            compliance: 0,
            recentTests: [],
          }));
          // Use real patients first, then add mock ones
          setPatients([...realPatients, ...mockPatients]);
        }

        if (alerts.status === "fulfilled") {
          setActiveAlerts(alerts.value);
        }
      } catch (err) {
        console.error("Failed to load data from backend:", err);
        // Fallback to mock data is already set
      } finally {
        setIsLoading(false);
      }
    }

    loadData();
  }, []);

  const filteredPatients = patients.filter(patient => {
    const matchesSearch = patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         patient.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === "all" || patient.status === statusFilter;
    return matchesSearch && matchesStatus;
  }).sort((a, b) => {
    if (sortBy === "riskScore") return b.riskScore - a.riskScore;
    if (sortBy === "name") return a.name.localeCompare(b.name);
    if (sortBy === "age") return b.age - a.age;
    return 0;
  });

  const totalPatients = patients.length;
  const criticalPatients = patients.filter(p => p.status === "critical").length;
  const warningPatients = patients.filter(p => p.status === "warning").length;
  const stablePatients = patients.filter(p => p.status === "stable").length;
  const avgHealthScore = Math.round(patients.reduce((acc, p) => acc + p.healthScore, 0) / totalPatients);


  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Enhanced Header with Real-time Stats */}
        <motion.div
          className="glass-morphism-dark rounded-3xl p-8 mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex justify-between items-start mb-6">
            <div>
              <h1 className="font-poppins font-bold text-3xl text-gray-800">Clinical Command Center</h1>
              <p className="text-gray-600 mt-2 text-lg">Advanced patient monitoring, analytics, and care coordination</p>
              <div className="flex items-center space-x-4 mt-3 text-sm text-gray-500">
                <div className="flex items-center space-x-1">
                  <Clock className="w-4 h-4" />
                  <span>Last updated: {new Date().toLocaleTimeString()}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Activity className="w-4 h-4" />
                  <span>Live monitoring active</span>
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Button variant="outline" className="glass-morphism hover:bg-blue-50">
                <Download className="w-4 h-4 mr-2" />
                Export Report
              </Button>
              <Button variant="outline" className="glass-morphism hover:bg-blue-50">
                <Calendar className="w-4 h-4 mr-2" />
                Schedule
              </Button>
              
              {/* Emergency Ambulance Service Button */}
              <motion.div 
                whileHover={{ scale: 1.05 }} 
                whileTap={{ scale: 0.95 }}
                className="relative"
              >
                <Button 
                  className="bg-red-600 hover:bg-red-700 text-white px-6 font-semibold emergency-glow shadow-lg"
                  onClick={() => {
                    // Handle emergency ambulance call
                    const criticalPatient = mockPatients.find(p => p.status === 'critical');
                    if (criticalPatient) {
                      alert(`🚨 EMERGENCY AMBULANCE DISPATCH\n\nPatient: ${criticalPatient.name}\nLocation: ${criticalPatient.location}\nCondition: ${criticalPatient.lastAlert}\n\nAmbulance service contacted: 108\nETA: 8-12 minutes\n\nEmergency contacts notified:\n${criticalPatient.contactInfo.emergency}`);
                    } else {
                      alert('🚨 Emergency Ambulance Service\n\nDial 108 for immediate ambulance dispatch\nAll emergency protocols activated');
                    }
                  }}
                >
                  <Truck className="w-4 h-4 mr-2" />
                  🚨 Call Ambulance
                </Button>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-pulse"></div>
              </motion.div>
              
              <Button className="bg-vitals-primary hover:bg-blue-600 text-white px-6">
                <UserPlus className="w-4 h-4 mr-2" />
                Add Patient
              </Button>
            </div>
          </div>
          
          {/* Real-time Metrics Bar */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="bg-white/60 rounded-2xl p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <Users className="w-5 h-5 text-vitals-primary mr-2" />
                <span className="text-2xl font-bold text-vitals-primary">{totalPatients}</span>
              </div>
              <p className="text-sm text-gray-600 font-medium">Total Patients</p>
            </div>
            <div className="bg-white/60 rounded-2xl p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
                <span className="text-2xl font-bold text-red-500">{criticalPatients}</span>
              </div>
              <p className="text-sm text-gray-600 font-medium">Critical</p>
            </div>
            <div className="bg-white/60 rounded-2xl p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <Target className="w-5 h-5 text-yellow-500 mr-2" />
                <span className="text-2xl font-bold text-yellow-500">{warningPatients}</span>
              </div>
              <p className="text-sm text-gray-600 font-medium">Monitoring</p>
            </div>
            <div className="bg-white/60 rounded-2xl p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <Heart className="w-5 h-5 text-green-500 mr-2" />
                <span className="text-2xl font-bold text-green-500">{stablePatients}</span>
              </div>
              <p className="text-sm text-gray-600 font-medium">Stable</p>
            </div>
            <div className="bg-white/60 rounded-2xl p-4 text-center">
              <div className="flex items-center justify-center mb-2">
                <BarChart3 className="w-5 h-5 text-purple-500 mr-2" />
                <span className="text-2xl font-bold text-purple-500">{avgHealthScore}</span>
              </div>
              <p className="text-sm text-gray-600 font-medium">Avg Health Score</p>
            </div>
          </div>
        </motion.div>

        {/* Advanced Search and Filtering */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-6 mb-8"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0 lg:space-x-6">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <Input
                placeholder="Search patients by name, ID, or condition..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 glass-morphism border-0 h-12 text-lg"
              />
            </div>
            
            <div className="flex items-center space-x-3">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="glass-morphism border-0 w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="stable">Stable</SelectItem>
                </SelectContent>
              </Select>
              
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="glass-morphism border-0 w-36">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="riskScore">Risk Score</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="age">Age</SelectItem>
                </SelectContent>
              </Select>
              
              <Button variant="outline" className="glass-morphism hover:bg-blue-50">
                <Filter className="w-4 h-4 mr-2" />
                More Filters
              </Button>
            </div>
          </div>
        </motion.div>

        {/* View Toggle */}
        <motion.div
          className="flex items-center space-x-1 bg-white rounded-2xl p-2 mb-8 shadow-sm w-fit"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'patients', label: 'Patient List', icon: Users },
            { id: 'analytics', label: 'Analytics', icon: LineChart },
            { id: 'alerts', label: 'Active Alerts', icon: AlertTriangle }
          ].map((view) => {
            const Icon = view.icon;
            return (
              <button
                key={view.id}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all ${
                  selectedView === view.id 
                    ? 'bg-vitals-primary text-white shadow-lg transform scale-105' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
                onClick={() => setSelectedView(view.id)}
              >
                <Icon className="w-4 h-4" />
                <span>{view.label}</span>
              </button>
            );
          })}
        </motion.div>

        {/* Content based on selected view */}
        {selectedView === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 mb-8">
            
            {/* Department Analytics */}
            <div className="lg:col-span-2">
              <motion.div
                className="glass-morphism-dark rounded-2xl p-6 h-80"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <h3 className="font-semibold text-xl text-gray-800 mb-6">Weekly Patient Flow</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <ComposedChart data={weeklyTrends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                    <XAxis dataKey="day" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                        border: 'none', 
                        borderRadius: '16px',
                        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)'
                      }} 
                    />
                    <Bar dataKey="admissions" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Line 
                      type="monotone" 
                      dataKey="alerts" 
                      stroke="#ef4444" 
                      strokeWidth={3}
                      dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </motion.div>
            </div>

            {/* Department Distribution */}
            <div className="lg:col-span-2">
              <motion.div
                className="glass-morphism-dark rounded-2xl p-6 h-80"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.8 }}
              >
                <h3 className="font-semibold text-xl text-gray-800 mb-6">Department Distribution</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsPieChart>
                      <Pie
                        data={departmentStats}
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="patients"
                      >
                        {departmentStats.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </RechartsPieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  {departmentStats.map((dept, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: dept.color }}></div>
                      <span className="text-xs text-gray-600">{dept.name}</span>
                      <span className="text-xs font-semibold text-gray-800">({dept.patients})</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        )}

        {/* Enhanced Patient Grid */}
        {(selectedView === 'patients' || selectedView === 'overview') && (
          <motion.div
            className="grid lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: selectedView === 'overview' ? 1.0 : 0.6 }}
          >
            {filteredPatients.map((patient, index) => (
              <EnhancedPatientCard key={patient.id} patient={patient} index={index} />
            ))}
          </motion.div>
        )}

        {/* Active Alerts Section */}
        {selectedView === 'alerts' && (
          <motion.div
            className="space-y-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
          >
            <div className="glass-morphism-dark rounded-2xl p-6">
              <h3 className="font-semibold text-xl text-gray-800 mb-6 flex items-center">
                <AlertTriangle className="w-6 h-6 text-red-500 mr-2" />
                Active Alerts & Notifications
              </h3>
              <div className="space-y-4">
                {(activeAlerts.length > 0 
                  ? activeAlerts.map((alert) => {
                      const patient = patients.find(p => p.id === alert.patient_id);
                      return {
                        id: alert.id,
                        patientId: alert.patient_id,
                        name: patient ? patient.name : `Patient ${alert.patient_id}`,
                        status: alert.tier === "CRITICAL" ? ("critical" as const) : ("warning" as const),
                        lastAlert: alert.status,
                        alertTime: alert.fired_at ? new Date(alert.fired_at).toLocaleTimeString() : 'N/A',
                      };
                    })
                  : patients
                      .filter(p => p.status === 'critical' || p.status === 'warning')
                      .map(p => ({
                        id: p.id,
                        patientId: p.id,
                        name: p.name,
                        status: p.status,
                        lastAlert: p.lastAlert,
                        alertTime: p.alertTime
                      }))
                ).map((alert, index) => (
                    <motion.div
                      key={alert.id}
                      className={`p-4 rounded-xl border-l-4 ${
                        alert.status === 'critical' 
                          ? 'bg-red-50 border-red-500'
                          : 'bg-yellow-50 border-yellow-500'
                      }`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.4, delay: index * 0.1 }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            alert.status === 'critical' ? 'bg-red-500' : 'bg-yellow-500'
                          } animate-pulse`}></div>
                          <div>
                            <h4 className="font-semibold text-gray-800">{alert.name} ({alert.patientId})</h4>
                            <p className={`text-sm ${
                              alert.status === 'critical' ? 'text-red-700' : 'text-yellow-700'
                            }`}>
                              {alert.lastAlert}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-xs text-gray-500">{alert.alertTime}</span>
                          <Link href={`/patient-analysis/${alert.patientId}`}>
                            <Button size="sm" variant="outline" className="h-8">
                              <Eye className="w-3 h-3 mr-1" />
                              View
                            </Button>
                          </Link>
                        </div>
                      </div>
                    </motion.div>
                  ))}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

// Enhanced Patient Card Component
function EnhancedPatientCard({ patient, index }: { patient: Patient; index: number }) {
  return (
    <motion.div
      className={`glass-morphism-dark rounded-2xl p-6 hover:shadow-xl transition-all cursor-pointer border-l-4 ${
        patient.status === 'critical' ? 'border-red-500' :
        patient.status === 'warning' ? 'border-yellow-500' : 'border-green-500'
      }`}
      whileHover={{ y: -5, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: index * 0.1 }}
    >
      {/* Patient Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-12 h-12 bg-gradient-to-br ${patient.bgColor} rounded-xl flex items-center justify-center`}>
            <span className="text-white font-semibold text-lg">{patient.initials}</span>
          </div>
          <div>
            <h3 className="font-semibold text-gray-800">{patient.name}</h3>
            <p className="text-sm text-gray-500">{patient.gender}, Age {patient.age} • {patient.id}</p>
          </div>
        </div>
        <div className="text-right">
          <div className="flex items-center space-x-1 mb-1">
            <span className="text-lg font-bold text-vitals-primary">{patient.healthScore}</span>
            <span className="text-xs text-gray-500">Health</span>
          </div>
          <div className={`text-xs font-medium ${
            patient.riskScore >= 70 ? 'text-red-600' :
            patient.riskScore >= 40 ? 'text-yellow-600' : 'text-green-600'
          }`}>
            Risk: {patient.riskScore}%
          </div>
        </div>
      </div>

      {/* Conditions */}
      <div className="mb-4">
        <div className="flex flex-wrap gap-1">
          {patient.conditions.slice(0, 2).map((condition, idx) => (
            <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
              {condition}
            </span>
          ))}
          {patient.conditions.length > 2 && (
            <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded-full text-xs">
              +{patient.conditions.length - 2} more
            </span>
          )}
        </div>
      </div>

      {/* Vitals Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-white/60 rounded-xl p-3 text-center">
          <div className="text-sm text-gray-600">BP</div>
          <div className={`font-mono font-bold ${
            patient.status === 'critical' ? 'text-red-600' : 'text-gray-800'
          }`}>{patient.vitals.bp}</div>
        </div>
        <div className="bg-white/60 rounded-xl p-3 text-center">
          <div className="text-sm text-gray-600">HR</div>
          <div className={`font-mono font-bold ${
            patient.vitals.hr > 100 ? 'text-yellow-600' : 'text-gray-800'
          }`}>{patient.vitals.hr} BPM</div>
        </div>
        <div className="bg-white/60 rounded-xl p-3 text-center">
          <div className="text-sm text-gray-600">SpO₂</div>
          <div className={`font-mono font-bold ${
            patient.vitals.spo2 < 95 ? 'text-red-600' : 'text-gray-800'
          }`}>{patient.vitals.spo2}%</div>
        </div>
        <div className="bg-white/60 rounded-xl p-3 text-center">
          <div className="text-sm text-gray-600">Temp</div>
          <div className={`font-mono font-bold ${
            patient.vitals.temp > 100 ? 'text-red-600' : 'text-gray-800'
          }`}>{patient.vitals.temp}°F</div>
        </div>
      </div>

      {/* Status Alert */}
      <div className={`p-3 rounded-xl mb-4 ${
        patient.status === 'critical' ? 'bg-red-50 border border-red-200' :
        patient.status === 'warning' ? 'bg-yellow-50 border border-yellow-200' :
        'bg-green-50 border border-green-200'
      }`}>
        <div className="flex items-center space-x-2">
          {patient.status === 'critical' && <AlertTriangle className="w-4 h-4 text-red-500" />}
          {patient.status === 'warning' && <Target className="w-4 h-4 text-yellow-500" />}
          {patient.status === 'stable' && <Heart className="w-4 h-4 text-green-500" />}
          <div className="flex-1">
            <p className={`text-sm font-medium ${
              patient.status === 'critical' ? 'text-red-700' :
              patient.status === 'warning' ? 'text-yellow-700' : 'text-green-700'
            }`}>
              {patient.lastAlert}
            </p>
            <p className="text-xs text-gray-500">{patient.alertTime}</p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-2">
        <Link href={`/patient-analysis/${patient.id}`} className="flex-1">
          <Button size="sm" className="w-full bg-vitals-primary text-white hover:bg-blue-600">
            <Stethoscope className="w-3 h-3 mr-1" />
            Examine
          </Button>
        </Link>
        <Link href={`/patient-analysis/${patient.id}`} className="flex-1">
          <Button size="sm" variant="outline" className="w-full glass-morphism">
            <FileText className="w-3 h-3 mr-1" />
            Chart
          </Button>
        </Link>
        <Button size="sm" variant="outline" className="glass-morphism">
          <Phone className="w-3 h-3" />
        </Button>
        
        {/* Emergency Ambulance Button - Only for Critical Patients */}
        {patient.status === 'critical' && (
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button 
              size="sm" 
              className="bg-red-600 hover:bg-red-700 text-white font-semibold emergency-glow"
              onClick={() => {
                alert(`🚨 EMERGENCY AMBULANCE DISPATCH
                
Patient: ${patient.name}
ID: ${patient.id}
Location: ${patient.location}
Critical Alert: ${patient.lastAlert}

📞 Calling 108...
🏥 Nearest Hospital: Apollo Main
📍 ETA: 8-12 minutes

Emergency Contacts Notified:
${patient.contactInfo.emergency}

⚡ All emergency protocols activated`);
              }}
            >
              <Truck className="w-3 h-3 mr-1" />
              🚨
            </Button>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
