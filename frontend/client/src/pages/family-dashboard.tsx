import { motion } from "framer-motion";
import { useState } from "react";
import { 
  Activity, Heart, Droplets, Wind, Thermometer, Calendar, TrendingUp, TrendingDown, 
  AlertTriangle, Shield, Brain, Pill, Clock, Target, BarChart3, PieChart, Zap,
  User, Phone, MapPin, FileText, Settings, Download, Share2, Bell, ChevronDown,
  Users, Home, Stethoscope, LineChart
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Link } from "wouter";
import { 
  LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  AreaChart, Area, BarChart, Bar, PieChart as RechartsPieChart, Cell, Pie,
  RadialBarChart, RadialBar, ComposedChart
} from "recharts";

const familyMembers = [
  {
    id: "mom",
    name: "Mom",
    fullName: "Priya Sharma",
    age: 58,
    gender: "Female",
    status: "healthy" as const,
    avatar: "https://images.unsplash.com/photo-1582750433449-648ed127bb54?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&h=150",
    healthScore: 85,
    riskLevel: "Low",
    lastCheckup: "2024-01-20",
    nextAppointment: "2024-03-15",
    conditions: ["None"],
    vitals: {
      heartRate: { value: 78, icon: Heart, status: "Normal", color: "text-vitals-healthy", trend: "stable" },
      bloodPressure: { value: "120/80", icon: Droplets, status: "Optimal", color: "text-vitals-healthy", trend: "improving" },
      oxygen: { value: "99%", icon: Wind, status: "Excellent", color: "text-vitals-healthy", trend: "stable" },
      temperature: { value: "98.4°F", icon: Thermometer, status: "Normal", color: "text-vitals-healthy", trend: "stable" }
    },
    medications: [
      { name: "Vitamin D3", adherence: 92, status: "Good" },
      { name: "Calcium", adherence: 88, status: "Good" }
    ],
    recentActivity: [
      { activity: "Morning walk completed", time: "2 hours ago", type: "exercise" },
      { activity: "Vitamins taken", time: "8 hours ago", type: "medication" },
      { activity: "Blood pressure normal", time: "1 day ago", type: "vital" }
    ]
  },
  {
    id: "dad",
    name: "Dad", 
    fullName: "Rajesh Kumar",
    age: 62,
    gender: "Male",
    status: "warning" as const,
    avatar: "https://images.unsplash.com/photo-1607990281513-2c110a25bd8c?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&h=150",
    healthScore: 68,
    riskLevel: "Moderate",
    lastCheckup: "2024-01-10",
    nextAppointment: "2024-02-10",
    conditions: ["Hypertension", "Pre-diabetes"],
    vitals: {
      heartRate: { value: 95, icon: Heart, status: "Elevated", color: "text-vitals-warning", trend: "increasing" },
      bloodPressure: { value: "140/90", icon: Droplets, status: "High", color: "text-vitals-warning", trend: "concerning" },
      oxygen: { value: "98%", icon: Wind, status: "Good", color: "text-vitals-healthy", trend: "stable" },
      temperature: { value: "98.6°F", icon: Thermometer, status: "Normal", color: "text-vitals-healthy", trend: "stable" }
    },
    medications: [
      { name: "Amlodipine", adherence: 94, status: "Excellent" },
      { name: "Metformin", adherence: 89, status: "Good" },
      { name: "Aspirin", adherence: 76, status: "Needs Improvement" }
    ],
    recentActivity: [
      { activity: "BP elevated - 145/92", time: "1 hour ago", type: "alert" },
      { activity: "Medication taken", time: "4 hours ago", type: "medication" },
      { activity: "Doctor consultation scheduled", time: "1 day ago", type: "appointment" }
    ]
  }
];

const vitalTrendsData = {
  mom: {
    heartRate: [
      { time: "Mon", value: 72, target: 75 },
      { time: "Tue", value: 74, target: 75 },
      { time: "Wed", value: 78, target: 75 },
      { time: "Thu", value: 76, target: 75 },
      { time: "Fri", value: 79, target: 75 },
      { time: "Sat", value: 77, target: 75 },
      { time: "Sun", value: 78, target: 75 }
    ],
    bloodPressure: [
      { time: "Week 1", systolic: 125, diastolic: 82 },
      { time: "Week 2", systolic: 122, diastolic: 80 },
      { time: "Week 3", systolic: 120, diastolic: 78 },
      { time: "Week 4", systolic: 118, diastolic: 76 }
    ]
  },
  dad: {
    heartRate: [
      { time: "Mon", value: 88, target: 75 },
      { time: "Tue", value: 92, target: 75 },
      { time: "Wed", value: 95, target: 75 },
      { time: "Thu", value: 89, target: 75 },
      { time: "Fri", value: 94, target: 75 },
      { time: "Sat", value: 91, target: 75 },
      { time: "Sun", value: 95, target: 75 }
    ],
    bloodPressure: [
      { time: "Week 1", systolic: 145, diastolic: 92 },
      { time: "Week 2", systolic: 142, diastolic: 90 },
      { time: "Week 3", systolic: 140, diastolic: 89 },
      { time: "Week 4", systolic: 138, diastolic: 88 }
    ]
  }
};

export default function FamilyDashboard() {
  const [selectedMember, setSelectedMember] = useState("dad"); // Start with dad since he needs more attention
  const [selectedTimeframe, setSelectedTimeframe] = useState("week");
  
  const currentMember = familyMembers.find(member => member.id === selectedMember) || familyMembers[0];
  const vitalData = vitalTrendsData[selectedMember as keyof typeof vitalTrendsData];

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving": return <TrendingUp className="w-4 h-4 text-vitals-healthy" />;
      case "increasing": case "concerning": return <TrendingUp className="w-4 h-4 text-vitals-warning" />;
      case "critical": return <AlertTriangle className="w-4 h-4 text-vitals-critical" />;
      default: return <TrendingDown className="w-4 h-4 text-gray-400" />;
    }
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case "exercise": return Activity;
      case "medication": return Pill;
      case "vital": return BarChart3;
      case "alert": return AlertTriangle;
      case "appointment": return Calendar;
      default: return FileText;
    }
  };

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Enhanced Header */}
        <motion.div
          className="flex justify-between items-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <h1 className="font-poppins font-bold text-3xl text-gray-800">Family Health Center</h1>
            <p className="text-gray-600 mt-2">Comprehensive health monitoring and care coordination for your loved ones</p>
          </div>
          <div className="flex items-center space-x-4">
            <Link href="/patient-analysis">
              <Button variant="outline" className="glass-morphism hover:bg-blue-50">
                <BarChart3 className="w-4 h-4 mr-2" />
                Advanced Analytics
              </Button>
            </Link>
            <Link href="/emergency" data-testid="emergency-button">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button className="bg-vitals-critical text-white hover:bg-red-600 emergency-glow font-semibold px-6 py-3 rounded-xl">
                  🆘 Emergency
                </Button>
              </motion.div>
            </Link>
          </div>
        </motion.div>

        {/* Family Member Selector */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-6 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-xl text-gray-800">Select Family Member</h2>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Users className="w-4 h-4" />
              <span>{familyMembers.length} members monitored</span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {familyMembers.map((member, index) => (
              <motion.div
                key={member.id}
                className={`p-4 rounded-2xl cursor-pointer transition-all border-2 ${
                  selectedMember === member.id 
                    ? 'border-vitals-primary bg-blue-50 shadow-lg' 
                    : 'border-transparent bg-white/60 hover:bg-white/80 hover:shadow-md'
                }`}
                onClick={() => setSelectedMember(member.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <img
                      src={member.avatar}
                      alt={member.name}
                      className="w-16 h-16 rounded-2xl object-cover"
                    />
                    <div className={`absolute -bottom-1 -right-1 w-6 h-6 rounded-full border-2 border-white flex items-center justify-center ${
                      member.status === 'healthy' ? 'bg-green-500' :
                      member.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}>
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    </div>
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-lg text-gray-800">{member.fullName}</h3>
                      <div className="flex items-center space-x-2">
                        <div className="text-2xl font-bold text-vitals-primary">{member.healthScore}</div>
                        <div className="text-xs text-gray-500">Health Score</div>
                      </div>
                    </div>
                    <p className="text-gray-600">{member.gender}, Age {member.age}</p>
                    <div className="flex items-center justify-between mt-2">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                        member.riskLevel === 'Low' ? 'bg-green-100 text-green-700' :
                        member.riskLevel === 'Moderate' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {member.riskLevel} Risk
                      </span>
                      <span className="text-xs text-gray-500">Next: {new Date(member.nextAppointment).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Selected Patient Deep Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          
          {/* Patient Overview Card */}
          <motion.div
            className="glass-morphism-dark rounded-2xl p-6"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <div className="text-center mb-6">
              <img
                src={currentMember.avatar}
                alt={currentMember.name}
                className="w-24 h-24 rounded-2xl object-cover mx-auto mb-4 shadow-lg"
              />
              <h3 className="font-bold text-xl text-gray-800">{currentMember.fullName}</h3>
              <p className="text-gray-600">{currentMember.gender}, Age {currentMember.age}</p>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-white/60 rounded-xl">
                <span className="text-sm font-medium text-gray-700">Health Score</span>
                <div className="flex items-center space-x-2">
                  <div className="w-16 h-2 bg-gray-200 rounded-full">
                    <div 
                      className="h-2 bg-vitals-primary rounded-full transition-all duration-500"
                      style={{ width: `${currentMember.healthScore}%` }}
                    ></div>
                  </div>
                  <span className="text-lg font-bold text-vitals-primary">{currentMember.healthScore}</span>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="font-medium text-gray-800">Current Conditions</h4>
                <div className="flex flex-wrap gap-2">
                  {currentMember.conditions.map((condition, index) => (
                    <span 
                      key={index} 
                      className={`px-3 py-1 rounded-full text-xs font-medium ${
                        condition === 'None' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'
                      }`}
                    >
                      {condition}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Current Vitals */}
          <motion.div
            className="lg:col-span-2 glass-morphism-dark rounded-2xl p-6"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-semibold text-xl text-gray-800">Current Vital Signs</h3>
              <div className="flex items-center space-x-2">
                <Activity className="w-5 h-5 text-vitals-primary" />
                <span className="text-sm text-gray-600">Live monitoring</span>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              </div>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(currentMember.vitals).map(([key, vital]) => (
                <motion.div
                  key={key}
                  className="bg-white/60 rounded-2xl p-4 text-center hover:bg-white/80 transition-all"
                  whileHover={{ y: -2, scale: 1.02 }}
                >
                  <div className="w-12 h-12 bg-white/80 rounded-2xl flex items-center justify-center mb-3">
                    <vital.icon className={`w-6 h-6 ${vital.color}`} />
                  </div>
                  <div className={`font-mono text-xl font-bold ${vital.color}`}>
                    {vital.value}
                  </div>
                  <div className="text-sm text-gray-600 mb-2">
                    {key === "heartRate" && "Heart Rate"}
                    {key === "bloodPressure" && "Blood Pressure"}
                    {key === "oxygen" && "Oxygen"}
                    {key === "temperature" && "Temperature"}
                  </div>
                  <div className="flex items-center justify-center space-x-1">
                    <span className={`text-xs font-medium ${vital.color}`}>{vital.status}</span>
                    {getTrendIcon(vital.trend)}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Advanced Analytics Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          
          {/* Vital Trends Chart */}
          <motion.div
            className="glass-morphism-dark rounded-2xl p-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-semibold text-xl text-gray-800">Heart Rate Trends</h3>
              <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
                <SelectTrigger className="glass-morphism border-0 w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="week">Week</SelectItem>
                  <SelectItem value="month">Month</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart data={vitalData.heartRate}>
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
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 5 }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="target" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Medication Adherence */}
          <motion.div
            className="glass-morphism-dark rounded-2xl p-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.0 }}
          >
            <h3 className="font-semibold text-xl text-gray-800 mb-6">Medication Adherence</h3>
            
            <div className="space-y-4">
              {currentMember.medications.map((med, index) => (
                <div key={index} className="p-4 bg-white/60 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-800">{med.name}</h4>
                    <span className={`text-sm font-medium ${
                      med.status === 'Excellent' ? 'text-green-600' :
                      med.status === 'Good' ? 'text-blue-600' : 'text-yellow-600'
                    }`}>
                      {med.status}
                    </span>
                  </div>
                  <div className="relative">
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-500 ${
                          med.adherence >= 90 ? 'bg-green-500' :
                          med.adherence >= 75 ? 'bg-blue-500' : 'bg-yellow-500'
                        }`}
                        style={{ width: `${med.adherence}%` }}
                      ></div>
                    </div>
                    <span className="absolute right-0 top-4 text-sm font-bold text-gray-700">
                      {med.adherence}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Recent Activity & Alerts */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-6"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.2 }}
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-semibold text-xl text-gray-800">Recent Activity & Care Timeline</h3>
            <Button variant="outline" size="sm" className="glass-morphism">
              <Calendar className="w-4 h-4 mr-2" />
              View All
            </Button>
          </div>
          
          <div className="space-y-3">
            {currentMember.recentActivity.map((activity, index) => (
              <motion.div
                key={index}
                className={`flex items-center space-x-4 p-4 rounded-xl ${
                  activity.type === 'alert' ? 'bg-red-50 border border-red-200' :
                  activity.type === 'medication' ? 'bg-blue-50 border border-blue-200' :
                  activity.type === 'exercise' ? 'bg-green-50 border border-green-200' :
                  'bg-purple-50 border border-purple-200'
                }`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <div className="w-10 h-10 bg-white/80 rounded-xl flex items-center justify-center">
                  {(() => {
                    const IconComponent = getActivityIcon(activity.type);
                    const iconColor = activity.type === 'alert' ? 'text-red-500' :
                                     activity.type === 'medication' ? 'text-blue-500' :
                                     activity.type === 'exercise' ? 'text-green-500' :
                                     activity.type === 'appointment' ? 'text-purple-500' :
                                     'text-gray-600';
                    return <IconComponent className={`w-5 h-5 ${iconColor}`} />;
                  })()}
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-800">{activity.activity}</p>
                  <p className="text-sm text-gray-500">{activity.time}</p>
                </div>
                {activity.type === 'alert' && (
                  <AlertTriangle className="w-5 h-5 text-red-500" />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
