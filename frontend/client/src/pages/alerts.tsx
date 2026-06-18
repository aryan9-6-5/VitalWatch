import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { Bell, AlertTriangle, CheckCircle, Clock, Phone, MessageSquare, Settings, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { getActiveAlerts, acknowledgeAlert, type Alert as BackendAlert } from "@/lib/api";

interface Alert {
  id: string;
  type: "critical" | "warning" | "info";
  title: string;
  description: string;
  member: string;
  timestamp: Date;
  status: "active" | "acknowledged" | "resolved";
  vitals?: {
    type: string;
    value: string;
    threshold: string;
  };
}

const mockAlerts: Alert[] = [
  {
    id: "1",
    type: "critical",
    title: "High Blood Pressure Alert",
    description: "Dad's blood pressure reading of 165/95 exceeds the critical threshold",
    member: "dad",
    timestamp: new Date(Date.now() - 2 * 60 * 1000), // 2 minutes ago
    status: "active",
    vitals: { type: "Blood Pressure", value: "165/95", threshold: "140/90" }
  },
  {
    id: "2",
    type: "warning",
    title: "Elevated Heart Rate",
    description: "Dad's heart rate of 98 BPM is above normal range",
    member: "dad",
    timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
    status: "acknowledged",
    vitals: { type: "Heart Rate", value: "98 BPM", threshold: "60-90 BPM" }
  },
  {
    id: "3",
    type: "info",
    title: "Medication Reminder",
    description: "Time for Dad's evening blood pressure medication",
    member: "dad",
    timestamp: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
    status: "resolved"
  },
  {
    id: "4",
    type: "info",
    title: "Device Sync Complete",
    description: "Mom's vitals successfully synced from Omron device",
    member: "mom",
    timestamp: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
    status: "resolved"
  },
  {
    id: "5",
    type: "warning",
    title: "Missed Reading",
    description: "No blood pressure reading received for Dad in the last 4 hours",
    member: "dad",
    timestamp: new Date(Date.now() - 60 * 60 * 1000), // 1 hour ago
    status: "active"
  }
];

const alertSettings = [
  { id: "critical", label: "Critical Alerts", description: "Immediate health emergencies", enabled: true },
  { id: "warning", label: "Warning Alerts", description: "Health metrics outside normal range", enabled: true },
  { id: "medication", label: "Medication Reminders", description: "Medicine timing notifications", enabled: true },
  { id: "device", label: "Device Notifications", description: "Device connectivity and sync status", enabled: false },
  { id: "family", label: "Family Notifications", description: "Share alerts with family members", enabled: true }
];

export default function Alerts() {
  const [selectedFilter, setSelectedFilter] = useState("all");
  const [selectedMember, setSelectedMember] = useState("all");
  const [settings, setSettings] = useState(alertSettings);
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);

  // Fetch real alerts from the backend on mount
  useEffect(() => {
    async function loadAlerts() {
      try {
        const backendAlerts = await getActiveAlerts();
        if (backendAlerts.length > 0) {
          const realAlerts: Alert[] = backendAlerts.map((ba: BackendAlert, i: number) => ({
            id: ba.id,
            type: ba.tier === "CRITICAL" ? "critical" as const : ba.tier === "WARNING" ? "warning" as const : "info" as const,
            title: `${ba.tier} Alert`,
            description: `Alert for patient ${ba.patient_id} — ${ba.status}`,
            member: "patient",
            timestamp: ba.fired_at ? new Date(ba.fired_at) : new Date(),
            status: ba.acknowledged ? "acknowledged" as const : "active" as const,
          }));
          setAlerts([...realAlerts, ...mockAlerts]);
        }
      } catch (err) {
        console.error("Failed to load alerts from backend:", err);
      }
    }
    loadAlerts();
  }, []);

  const handleAcknowledge = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId);
      setAlerts(prev => prev.map(a => 
        a.id === alertId ? { ...a, status: "acknowledged" as const } : a
      ));
    } catch (err) {
      console.error("Failed to acknowledge alert:", err);
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "critical": return <AlertTriangle className="w-5 h-5 text-vitals-critical" />;
      case "warning": return <Bell className="w-5 h-5 text-vitals-warning" />;
      case "info": return <CheckCircle className="w-5 h-5 text-vitals-primary" />;
      default: return <Bell className="w-5 h-5 text-gray-400" />;
    }
  };

  const getAlertBorder = (type: string) => {
    switch (type) {
      case "critical": return "border-l-vitals-critical";
      case "warning": return "border-l-vitals-warning";
      case "info": return "border-l-vitals-primary";
      default: return "border-l-gray-300";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active": return { bg: "bg-red-100", text: "text-red-700", label: "Active" };
      case "acknowledged": return { bg: "bg-yellow-100", text: "text-yellow-700", label: "Acknowledged" };
      case "resolved": return { bg: "bg-green-100", text: "text-green-700", label: "Resolved" };
      default: return { bg: "bg-gray-100", text: "text-gray-700", label: "Unknown" };
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    if (selectedFilter !== "all" && alert.type !== selectedFilter) return false;
    if (selectedMember !== "all" && alert.member !== selectedMember) return false;
    return true;
  });

  const toggleSetting = (id: string) => {
    setSettings(prev => prev.map(setting => 
      setting.id === id ? { ...setting, enabled: !setting.enabled } : setting
    ));
  };

  const formatTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / (1000 * 60));
    
    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes} min ago`;
    
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    
    const days = Math.floor(hours / 24);
    return `${days} day${days > 1 ? 's' : ''} ago`;
  };

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-red-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          className="flex justify-between items-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <h1 className="font-poppins font-bold text-3xl text-gray-800">Smart Alerts & Escalation</h1>
            <p className="text-gray-600 mt-2">Intelligent health monitoring with automated family notifications</p>
          </div>
          <div className="flex items-center space-x-4">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button variant="outline" className="glass-morphism hover:bg-blue-50" data-testid="alert-settings">
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
            </motion.div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button className="bg-vitals-primary text-white hover:bg-blue-600" data-testid="emergency-contacts">
                <Phone className="w-4 h-4 mr-2" />
                Emergency Contacts
              </Button>
            </motion.div>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Alerts List */}
          <div className="lg:col-span-2 space-y-6">
            {/* Filters */}
            <motion.div
              className="flex items-center space-x-4 flex-wrap gap-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="flex items-center space-x-2">
                <Filter className="w-4 h-4 text-gray-600" />
                <span className="text-sm font-medium text-gray-700">Filter by:</span>
              </div>
              <Select value={selectedFilter} onValueChange={setSelectedFilter}>
                <SelectTrigger className="glass-morphism border-0 w-32" data-testid="filter-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="info">Info</SelectItem>
                </SelectContent>
              </Select>
              <Select value={selectedMember} onValueChange={setSelectedMember}>
                <SelectTrigger className="glass-morphism border-0 w-32" data-testid="filter-member">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Members</SelectItem>
                  <SelectItem value="mom">Mom</SelectItem>
                  <SelectItem value="dad">Dad</SelectItem>
                </SelectContent>
              </Select>
            </motion.div>

            {/* Alert Cards */}
            <div className="space-y-4">
              {filteredAlerts.map((alert, index) => {
                const statusBadge = getStatusBadge(alert.status);
                return (
                  <motion.div
                    key={alert.id}
                    className={`glass-morphism-dark rounded-2xl p-6 border-l-4 ${getAlertBorder(alert.type)} hover:shadow-xl transition-all`}
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                    whileHover={{ y: -2 }}
                    data-testid={`alert-${alert.id}`}
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-start space-x-3">
                        {getAlertIcon(alert.type)}
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="font-semibold text-gray-800">{alert.title}</h3>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusBadge.bg} ${statusBadge.text}`}>
                              {statusBadge.label}
                            </span>
                          </div>
                          <p className="text-gray-600 text-sm mb-2">{alert.description}</p>
                          <div className="flex items-center space-x-4 text-xs text-gray-500">
                            <span className="flex items-center space-x-1">
                              <Clock className="w-3 h-3" />
                              <span>{formatTimeAgo(alert.timestamp)}</span>
                            </span>
                            <span className="capitalize">{alert.member}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {alert.vitals && (
                      <div className="bg-gray-50 rounded-xl p-4 mb-4">
                        <div className="grid grid-cols-3 gap-4 text-center">
                          <div>
                            <div className="text-xs text-gray-500">Vital Type</div>
                            <div className="font-semibold text-gray-800">{alert.vitals.type}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Current Value</div>
                            <div className="font-semibold text-vitals-critical">{alert.vitals.value}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Threshold</div>
                            <div className="font-semibold text-gray-800">{alert.vitals.threshold}</div>
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="flex space-x-3">
                      {alert.status === "active" && (
                        <>
                          <Button size="sm" className="bg-vitals-primary text-white hover:bg-blue-600" data-testid={`acknowledge-${alert.id}`} onClick={() => handleAcknowledge(alert.id)}>
                            Acknowledge
                          </Button>
                          {alert.type === "critical" && (
                            <Button size="sm" variant="outline" className="border-vitals-critical text-vitals-critical hover:bg-red-50" data-testid={`escalate-${alert.id}`}>
                              <Phone className="w-3 h-3 mr-1" />
                              Call Doctor
                            </Button>
                          )}
                        </>
                      )}
                      <Button size="sm" variant="outline" className="hover:bg-blue-50" data-testid={`contact-family-${alert.id}`}>
                        <MessageSquare className="w-3 h-3 mr-1" />
                        Notify Family
                      </Button>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Settings Panel */}
          <motion.div
            className="space-y-6"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            {/* Alert Statistics */}
            <div className="glass-morphism-dark rounded-2xl p-6">
              <h3 className="font-semibold text-lg text-gray-800 mb-4">Alert Overview</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Active Alerts</span>
                  <span className="text-lg font-bold text-vitals-critical">
                    {alerts.filter(a => a.status === "active").length}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">This Week</span>
                  <span className="text-lg font-bold text-gray-800">12</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Response Time</span>
                  <span className="text-lg font-bold text-vitals-healthy">2.3 min</span>
                </div>
              </div>
            </div>

            {/* Notification Settings */}
            <div className="glass-morphism-dark rounded-2xl p-6">
              <h3 className="font-semibold text-lg text-gray-800 mb-4">Notification Settings</h3>
              <div className="space-y-4">
                {settings.map((setting) => (
                  <div key={setting.id} className="flex items-center justify-between p-3 bg-white/50 rounded-xl">
                    <div className="flex-1">
                      <div className="font-medium text-gray-800">{setting.label}</div>
                      <div className="text-xs text-gray-500">{setting.description}</div>
                    </div>
                    <Switch
                      checked={setting.enabled}
                      onCheckedChange={() => toggleSetting(setting.id)}
                      data-testid={`setting-${setting.id}`}
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Emergency Escalation */}
            <div className="glass-morphism-dark rounded-2xl p-6 border-2 border-vitals-critical/20">
              <h3 className="font-semibold text-lg text-gray-800 mb-4 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-vitals-critical" />
                Emergency Escalation
              </h3>
              <div className="space-y-3">
                <div className="text-sm text-gray-700">
                  <strong>Level 1:</strong> Family notification via app & SMS
                </div>
                <div className="text-sm text-gray-700">
                  <strong>Level 2:</strong> Doctor notification + WhatsApp alert
                </div>
                <div className="text-sm text-gray-700">
                  <strong>Level 3:</strong> Emergency services + GPS location
                </div>
              </div>
              <Button className="w-full mt-4 bg-vitals-critical text-white hover:bg-red-600" data-testid="test-escalation">
                Test Emergency System
              </Button>
            </div>

            {/* Quick Actions */}
            <div className="glass-morphism-dark rounded-2xl p-6">
              <h3 className="font-semibold text-lg text-gray-800 mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <Button variant="outline" className="w-full justify-start hover:bg-blue-50" data-testid="silence-all">
                  <Bell className="w-4 h-4 mr-2" />
                  Silence All Alerts (1 hour)
                </Button>
                <Button variant="outline" className="w-full justify-start hover:bg-green-50" data-testid="mark-all-read">
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Mark All as Read
                </Button>
                <Button variant="outline" className="w-full justify-start hover:bg-purple-50" data-testid="export-history">
                  <Settings className="w-4 h-4 mr-2" />
                  Export Alert History
                </Button>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
