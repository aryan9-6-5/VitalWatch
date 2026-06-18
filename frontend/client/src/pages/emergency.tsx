import { motion } from "framer-motion";
import { MapPin, HeartPulse, Zap, PhoneCall, UserCheck, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import EmergencyButton from "@/components/EmergencyButton";

export default function Emergency() {
  const emergencyContacts = [
    {
      name: "Dr. Rajesh Kumar",
      role: "Family Doctor",
      phone: "+91 98765 43210",
      icon: UserCheck,
      color: "bg-blue-500"
    },
    {
      name: "Emergency Services",
      role: "108 - Medical Emergency",
      phone: "108",
      icon: PhoneCall,
      color: "bg-red-500"
    },
    {
      name: "Family Group",
      role: "WhatsApp • 4 members",
      phone: "WhatsApp",
      icon: Users,
      color: "bg-green-500"
    }
  ];

  const currentVitals = [
    { label: "Heart Rate", value: "78 BPM", color: "text-vitals-healthy" },
    { label: "Blood Pressure", value: "120/80", color: "text-vitals-healthy" },
    { label: "Oxygen", value: "99%", color: "text-vitals-healthy" },
    { label: "Temperature", value: "98.6°F", color: "text-vitals-healthy" }
  ];

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-red-50 to-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Emergency Header */}
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.div
            className="w-24 h-24 bg-vitals-critical rounded-full flex items-center justify-center mx-auto mb-6 emergency-glow"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <PhoneCall className="w-12 h-12 text-white" />
          </motion.div>
          <h1 className="font-poppins font-bold text-4xl text-vitals-critical mb-4">Emergency Response</h1>
          <p className="text-xl text-gray-600">Immediate help when you need it most</p>
        </motion.div>

        {/* Emergency Actions */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Quick Emergency */}
          <motion.div
            className="glass-morphism-dark rounded-3xl p-8 border-2 border-vitals-critical hover:shadow-2xl transition-all"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-vitals-critical rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-poppins font-bold text-2xl text-gray-800 mb-2">Quick Alert</h3>
              <p className="text-gray-600">Send GPS location + health snapshot to all family members</p>
            </div>

            <Button 
              className="w-full bg-vitals-critical text-white py-4 rounded-xl font-bold text-lg hover:bg-red-600 transition-all emergency-glow"
              data-testid="quick-alert-button"
            >
              🚨 Send Family Alert
            </Button>

            <div className="mt-4 text-center text-sm text-gray-500">
              One tap to notify all family members
            </div>
          </motion.div>

          {/* Critical Emergency */}
          <motion.div
            className="glass-morphism-dark rounded-3xl p-8 border-2 border-red-600 hover:shadow-2xl transition-all"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <div className="text-center mb-6">
              <motion.div
                className="w-16 h-16 bg-red-600 rounded-2xl flex items-center justify-center mx-auto mb-4"
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <PhoneCall className="w-8 h-8 text-white" />
              </motion.div>
              <h3 className="font-poppins font-bold text-2xl text-gray-800 mb-2">Critical Emergency</h3>
              <p className="text-gray-600">Alert ambulance, clinic, and all caretakers immediately</p>
            </div>

            <EmergencyButton />

            <div className="mt-4 text-center text-sm text-gray-500">
              Triple-tap to activate emergency response
            </div>
          </motion.div>
        </div>

        {/* Current Status */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-6 mb-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <h3 className="font-semibold text-lg text-gray-800 mb-4 flex items-center">
            <MapPin className="w-5 h-5 mr-2 text-vitals-primary" />
            Current Location & Status
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <div className="text-sm text-gray-600 mb-1">Location</div>
              <div className="font-medium text-gray-800">Koramangala, Bangalore, Karnataka</div>
              <div className="text-sm text-gray-500">Accuracy: ±5 meters</div>
            </div>
            <div>
              <div className="text-sm text-gray-600 mb-1">Emergency Contacts</div>
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Dr. Rajesh Kumar</span>
                  <span className="text-xs text-vitals-healthy">✓ Available</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Family Group</span>
                  <span className="text-xs text-vitals-healthy">✓ 4 members</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Recent Health Snapshot */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-6 mb-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          <h3 className="font-semibold text-lg text-gray-800 mb-4 flex items-center">
            <HeartPulse className="w-5 h-5 mr-2 text-vitals-critical" />
            Current Health Snapshot
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {currentVitals.map((vital) => (
              <div key={vital.label} className="text-center bg-white/50 rounded-xl p-3">
                <div className={`font-mono text-lg font-bold ${vital.color}`}>{vital.value}</div>
                <div className="text-xs text-gray-500">{vital.label}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 text-center text-sm text-gray-500">
            Last updated: 2 minutes ago
          </div>
        </motion.div>

        {/* Emergency Contacts */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-6"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1 }}
        >
          <h3 className="font-semibold text-lg text-gray-800 mb-4">Emergency Contacts</h3>
          <div className="space-y-4">
            {emergencyContacts.map((contact, index) => (
              <motion.div
                key={contact.name}
                className="flex items-center justify-between p-4 bg-white/50 rounded-xl"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 1.2 + index * 0.1 }}
                whileHover={{ scale: 1.02 }}
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-12 h-12 ${contact.color} rounded-full flex items-center justify-center`}>
                    <contact.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <div className="font-medium">{contact.name}</div>
                    <div className="text-sm text-gray-500">{contact.role}</div>
                  </div>
                </div>
                <Button 
                  className={`${contact.color} text-white hover:opacity-90 transition-colors px-4 py-2 rounded-lg`}
                  data-testid={`call-${contact.name.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  {contact.phone === "WhatsApp" ? "Message" : "Call"}
                </Button>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
