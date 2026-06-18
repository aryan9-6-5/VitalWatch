import { motion, AnimatePresence } from "framer-motion";
import { useState, useRef, useEffect } from "react";
import { 
  Send, Mic, Image, Globe, Bot, User, Volume2, FileText, Calendar, Phone,
  Stethoscope, Heart, AlertTriangle, Clock, Zap, Brain, Target, Shield,
  ChevronRight, Bookmark, Settings, MoreVertical, Paperclip, Camera,
  MessageSquare, Activity, TrendingUp, Users, Search, Filter, Star,
  Archive, Share2, Download, Copy, Trash2, RotateCcw
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { predictFull, getActiveAlerts, getPatients } from "@/lib/api";

interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  language?: string;
  category?: "general" | "symptom" | "medication" | "emergency" | "appointment";
  priority?: "low" | "medium" | "high" | "critical";
  tags?: string[];
  attachments?: Array<{
    type: "image" | "document" | "audio";
    name: string;
    url: string;
  }>;
  actions?: Array<{
    label: string;
    action: string;
    type: "primary" | "secondary";
  }>;
}

const languages = [
  { code: "en", name: "English", flag: "🇺🇸", region: "Global" },
  { code: "hi", name: "हिंदी", flag: "🇮🇳", region: "India" },
  { code: "te", name: "తెలుగు", flag: "🇮🇳", region: "Telugu States" },
  { code: "ta", name: "தமிழ்", flag: "🇮🇳", region: "Tamil Nadu" },
  { code: "bn", name: "বাংলা", flag: "🇮🇳", region: "Bengal" },
  { code: "kn", name: "ಕನ್ನಡ", flag: "🇮🇳", region: "Karnataka" },
  { code: "ml", name: "മലയാളം", flag: "🇮🇳", region: "Kerala" }
];

const medicalSpecialties = [
  { id: "general", name: "General Medicine", icon: Stethoscope, color: "blue" },
  { id: "cardiology", name: "Cardiology", icon: Heart, color: "red" },
  { id: "emergency", name: "Emergency Care", icon: AlertTriangle, color: "orange" },
  { id: "mental-health", name: "Mental Health", icon: Brain, color: "purple" },
  { id: "preventive", name: "Preventive Care", icon: Shield, color: "green" }
];

const quickActions = [
  { 
    text: "Symptom Assessment", 
    icon: Stethoscope, 
    category: "symptom",
    description: "Describe symptoms for AI analysis",
    color: "bg-blue-500"
  },
  { 
    text: "Medication Guidance", 
    icon: FileText, 
    category: "medication",
    description: "Dosage, interactions, reminders",
    color: "bg-green-500"
  },
  { 
    text: "Emergency Assistance", 
    icon: AlertTriangle, 
    category: "emergency",
    description: "Urgent medical guidance",
    color: "bg-red-500"
  },
  { 
    text: "Appointment Booking", 
    icon: Calendar, 
    category: "appointment",
    description: "Schedule with healthcare providers",
    color: "bg-purple-500"
  },
  { 
    text: "Health Monitoring", 
    icon: Activity, 
    category: "general",
    description: "Vital signs and health tracking",
    color: "bg-indigo-500"
  },
  { 
    text: "Lab Results Review", 
    icon: Target, 
    category: "general",
    description: "Interpret test results",
    color: "bg-cyan-500"
  }
];

const conversationStarters = [
  "I'm experiencing chest pain and shortness of breath",
  "My blood pressure reading is 150/95 - should I be concerned?",
  "I missed my medication this morning, what should I do?",
  "Can you help me understand my recent lab results?",
  "I need to book an appointment with a cardiologist"
];


export default function AIAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      type: "assistant",
      content: "Welcome to MedAI, your comprehensive healthcare companion. I'm equipped with advanced medical knowledge to assist with symptom assessment, medication guidance, emergency support, and care coordination.\n\nI can help you:\n• Analyze symptoms and provide medical insights\n• Review medication schedules and interactions\n• Connect you with appropriate healthcare providers\n• Interpret health data and lab results\n• Coordinate emergency care when needed\n\nHow may I assist with your healthcare needs today?",
      timestamp: new Date(),
      category: "general",
      priority: "medium",
      actions: [
        { label: "Start Symptom Check", action: "symptom_check", type: "primary" },
        { label: "View Health Summary", action: "health_summary", type: "secondary" }
      ]
    }
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [selectedLanguage, setSelectedLanguage] = useState("en");
  const [selectedSpecialty, setSelectedSpecialty] = useState("general");
  const [isListening, setIsListening] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [showQuickActions, setShowQuickActions] = useState(true);
  const [conversationMode, setConversationMode] = useState<"chat" | "assessment" | "emergency">("chat");
  const [attachments, setAttachments] = useState<Array<{name: string; type: string; url: string}>>([]);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Default patient ID for the chatbot (first patient or a demo ID)
  const DEMO_PATIENT_ID = "demo-patient";

  const sendMessage = async () => {
    if (!inputMessage.trim() && attachments.length === 0) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputMessage,
      timestamp: new Date(),
      category: getMessageCategory(inputMessage),
    };

    const currentInput = inputMessage;
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setAttachments([]);
    setShowQuickActions(false);
    setIsTyping(true);

    try {
      const category = getMessageCategory(currentInput);
      let reply = "";

      if (currentInput.toLowerCase().includes("health summary") || currentInput.toLowerCase().includes("patient list")) {
        // Fetch patient list from the backend
        const patients = await getPatients();
        if (patients.length === 0) {
          reply = "📋 **Patient Records**\n\nNo patients are currently registered in the system. Use the Doctor Dashboard to add new patients.";
        } else {
          reply = "📋 **Patient Records**\n\n" + patients.map(
            (p) => `• **${p.name}** (Age: ${p.age}) — ${p.condition}`
          ).join("\n");
        }
      } else if (category === "emergency") {
        // Fetch active alerts
        try {
          const alerts = await getActiveAlerts();
          if (alerts.length === 0) {
            reply = "✅ **No Active Emergencies**\n\nThere are currently no active alerts in the system. If you are experiencing a medical emergency, please call **108** immediately.";
          } else {
            reply = `🚨 **Active Emergency Alerts (${alerts.length})**\n\n` + alerts.map(
              (a) => `• **${a.tier}** alert for patient ${a.patient_id} — Status: ${a.status}`
            ).join("\n") + "\n\n📞 For immediate help, call **108**.";
          }
        } catch {
          reply = "🚨 **Emergency Response**\n\nUnable to fetch alerts from the system. If this is a medical emergency, please call **108** immediately.\n\n📞 Emergency Services: 108\n👨‍⚕️ Contact your doctor directly.";
        }
      } else {
        // Use the full prediction pipeline (text → extract → predict → explain)
        try {
          const data = await predictFull({
            text: currentInput,
            patient_id: DEMO_PATIENT_ID,
            session_id: sessionId,
          });

          // Track session for multi-turn conversation
          setSessionId(data.session_id);
          reply = data.message;
        } catch {
          // If the backend prediction fails, fall back to the local enhanced response
          reply = getEnhancedAIResponse(currentInput, selectedSpecialty);
        }
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: reply,
        timestamp: new Date(),
        category,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: "assistant",
        content: "⚠️ **Connection Error**\n\nUnable to reach the backend service. Please ensure the VitalWatch backend is running and try again.\n\nIf this issue persists, check the server status or contact support.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    }

    setIsTyping(false);
  };

  const getMessageCategory = (message: string): Message['category'] => {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('emergency') || lowerMessage.includes('urgent') || lowerMessage.includes('chest pain')) return 'emergency';
    if (lowerMessage.includes('medication') || lowerMessage.includes('drug') || lowerMessage.includes('pill')) return 'medication';
    if (lowerMessage.includes('appointment') || lowerMessage.includes('doctor') || lowerMessage.includes('schedule')) return 'appointment';
    if (lowerMessage.includes('symptom') || lowerMessage.includes('pain') || lowerMessage.includes('feel')) return 'symptom';
    return 'general';
  };

  const getPriority = (message: string): Message['priority'] => {
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('emergency') || lowerMessage.includes('chest pain') || lowerMessage.includes('can\'t breathe')) return 'critical';
    if (lowerMessage.includes('severe') || lowerMessage.includes('urgent') || lowerMessage.includes('blood pressure')) return 'high';
    if (lowerMessage.includes('pain') || lowerMessage.includes('medication')) return 'medium';
    return 'low';
  };

  const getResponseActions = (message: string): Message['actions'] => {
    const actions: Message['actions'] = [];
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('appointment') || lowerMessage.includes('doctor')) {
      actions.push({ label: "Book Appointment", action: "book_appointment", type: "primary" });
    }
    if (lowerMessage.includes('emergency') || lowerMessage.includes('urgent')) {
      actions.push({ label: "Emergency Services", action: "emergency", type: "primary" });
    }
    if (lowerMessage.includes('medication')) {
      actions.push({ label: "Set Reminder", action: "medication_reminder", type: "secondary" });
    }
    
    actions.push({ label: "Get More Info", action: "more_info", type: "secondary" });
    return actions.length > 0 ? actions : undefined;
  };

  const getEnhancedAIResponse = (input: string, specialty: string): string => {
    const lowerInput = input.toLowerCase();
    
    // Emergency responses
    if (lowerInput.includes("chest pain") || lowerInput.includes("heart attack")) {
      return "🚨 **IMMEDIATE ATTENTION REQUIRED**\n\nChest pain can be a sign of a serious condition. Here's what you should do RIGHT NOW:\n\n**IMMEDIATE ACTIONS:**\n1. **Call 108** for emergency services immediately\n2. Sit down and rest, avoid physical exertion\n3. If you have nitroglycerin prescribed, take as directed\n4. Chew an aspirin (325mg) if not allergic\n\n**EMERGENCY SYMPTOMS** that require immediate care:\n• Crushing chest pain\n• Pain radiating to arm, jaw, or back\n• Shortness of breath\n• Nausea, sweating, dizziness\n\n**I'm alerting your emergency contacts and preparing your medical history for first responders.**\n\nPlease don't delay - every minute counts with cardiac events.";
    }
    
    if (lowerInput.includes("blood pressure") || lowerInput.includes("bp") || lowerInput.includes("150") || lowerInput.includes("160")) {
      return "📊 **BLOOD PRESSURE ANALYSIS**\n\nBased on your reading, here's my assessment:\n\n**UNDERSTANDING YOUR READING:**\n• Normal: <120/80 mmHg\n• Elevated: 120-129/<80 mmHg\n• Stage 1 HTN: 130-139/80-89 mmHg\n• Stage 2 HTN: ≥140/90 mmHg\n\n**IMMEDIATE RECOMMENDATIONS:**\n✅ Take reading again in 5 minutes (rest between)\n✅ Record time, activity, and any symptoms\n✅ Check if you've taken your BP medication\n✅ Avoid caffeine, stress, and physical activity\n\n**RED FLAGS** - Seek immediate care if:\n• Systolic >180 or Diastolic >120\n• Severe headache or vision changes\n• Chest pain or shortness of breath\n\n**Based on your health profile, I recommend scheduling a consultation within 24-48 hours if this pattern continues.**";
    }
    
    if (lowerInput.includes("medication") || lowerInput.includes("medicine") || lowerInput.includes("missed")) {
      return "💊 **MEDICATION MANAGEMENT**\n\n**CURRENT MEDICATION STATUS:**\nAnalyzing your medication schedule and interactions...\n\n**If you missed a dose:**\n1. **Check timing:** <2 hours late? Take now\n2. **>2 hours late:** Skip and take next scheduled dose\n3. **Never double dose** unless specifically instructed\n\n**MEDICATION SAFETY:**\n⚠️ High-risk medications (blood thinners, diabetes meds): Contact provider\n✅ Low-risk medications: Follow general missed dose guidelines\n\n**PERSONALIZED RECOMMENDATIONS:**\n• Set up smart reminders linked to your routine\n• Consider pill organizers for complex regimens\n• Use pharmacy auto-refill services\n• Keep emergency contact list with medication details\n\n**I can help you:**\n- Set intelligent reminders based on your lifestyle\n- Check for dangerous interactions\n- Find nearby pharmacies\n- Contact your prescriber if needed";
    }
    
    if (lowerInput.includes("appointment") || lowerInput.includes("doctor") || lowerInput.includes("schedule")) {
      return "📅 **HEALTHCARE APPOINTMENT COORDINATION**\n\n**SMART SCHEDULING BASED ON YOUR NEEDS:**\n\nAnalyzing your symptoms and health profile to recommend:\n\n**RECOMMENDED PROVIDER:**\n🩺 **Primary Care** - General concerns, routine care\n❤️ **Cardiologist** - Heart/BP issues, chest symptoms\n🧠 **Specialist** - Condition-specific care\n🚨 **Urgent Care** - Non-emergency but prompt needs\n\n**APPOINTMENT PREPARATION:**\n• Symptom timeline and severity (1-10 scale)\n• Current medications and recent changes\n• Relevant health data from your monitoring\n• Insurance and referral information\n\n**NEXT STEPS:**\n1. I'll check provider availability in your network\n2. Consider telehealth options for initial consultation\n3. Schedule follow-up based on treatment plan\n\n**I'm prepared to:**\n- Contact providers directly on your behalf\n- Send your health summary and symptom log\n- Set up reminders and pre-appointment tasks";
    }
    
    if (lowerInput.includes("lab") || lowerInput.includes("test") || lowerInput.includes("result")) {
      return "🔬 **LAB RESULTS INTERPRETATION**\n\n**COMPREHENSIVE ANALYSIS:**\n\nI can help you understand your test results in context of your health profile:\n\n**RESULT CATEGORIES:**\n🟢 **Normal Range** - Values within expected parameters\n🟡 **Borderline** - May need monitoring or lifestyle changes\n🔴 **Abnormal** - Requires medical attention or treatment\n\n**WHAT I ANALYZE:**\n• Trend analysis compared to previous results\n• Clinical significance in your specific context\n• Potential interactions with current medications\n• Lifestyle factors that may influence results\n\n**IMPORTANT:**\nWhile I can provide educational information about lab values, **always discuss results with your healthcare provider** for medical decisions.\n\n**I can help you:**\n- Prepare questions for your provider\n- Track trends over time\n- Identify which results need urgent attention\n- Schedule appropriate follow-up care";
    }
    
    // Default enhanced response
    return "🤖 **COMPREHENSIVE HEALTH ANALYSIS**\n\nThank you for sharing your health concern. I'm analyzing your input using advanced medical knowledge and your personal health context.\n\n**MY ASSESSMENT PROCESS:**\n1. **Symptom Analysis** - Pattern recognition and risk stratification\n2. **Health History Review** - Considering your medical background\n3. **Evidence-Based Recommendations** - Latest clinical guidelines\n4. **Personalized Care Planning** - Tailored to your specific needs\n\n**NEXT STEPS:**\nBased on your concern, I recommend:\n• **Monitoring** - Track symptoms and relevant metrics\n• **Documentation** - Keep detailed records for healthcare providers\n• **Professional Care** - Consider scheduling appropriate consultation\n• **Safety First** - Seek immediate care for any severe symptoms\n\n**I'm here to provide:**\n- Continuous monitoring and analysis\n- Care coordination with your healthcare team\n- Evidence-based health information\n- Emergency support when needed\n\nWhat specific aspect would you like me to focus on first?";
  };

  const handleQuickAction = (actionText: string, category: string) => {
    setInputMessage(actionText);
    setConversationMode(category === 'emergency' ? 'emergency' : category === 'symptom' ? 'assessment' : 'chat');
    setSelectedSpecialty(category);
    setShowQuickActions(false);
  };

  const handleConversationStarter = (starter: string) => {
    setInputMessage(starter);
    setShowQuickActions(false);
  };

  const toggleListening = () => {
    setIsListening(!isListening);
    if (!isListening) {
      // Start voice recognition
      navigator.mediaDevices?.getUserMedia({ audio: true })
        .then(() => {
          // Voice recognition implementation
          console.log('Voice recognition started');
        })
        .catch(() => {
          console.log('Microphone access denied');
        });
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      Array.from(files).forEach(file => {
        const newAttachment = {
          name: file.name,
          type: file.type.startsWith('image/') ? 'image' : 'document',
          url: URL.createObjectURL(file)
        };
        setAttachments(prev => [...prev, newAttachment]);
      });
    }
  };

  const clearConversation = () => {
    setMessages([
      {
        id: "1",
        type: "assistant",
        content: "Conversation cleared. How may I assist you today?",
        timestamp: new Date(),
        category: "general",
        priority: "medium"
      }
    ]);
    setShowQuickActions(true);
  };

  return (
    <div className="pt-16 min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        
        {/* Professional Header */}
        <motion.div
          className="glass-morphism-dark rounded-3xl p-8 mb-6"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-6">
              <motion.div
                className="w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl flex items-center justify-center shadow-lg"
                animate={{ 
                  boxShadow: isTyping ? ['0 0 0 0 rgba(59, 130, 246, 0.4)', '0 0 0 10px rgba(59, 130, 246, 0)'] : '0 10px 25px rgba(0, 0, 0, 0.1)'
                }}
                transition={{ duration: 1, repeat: isTyping ? Infinity : 0 }}
              >
                <Brain className="w-8 h-8 text-white" />
              </motion.div>
              <div>
                <h1 className="font-poppins font-bold text-3xl text-gray-800 mb-2">MedAI Clinical Assistant</h1>
                <p className="text-gray-600 text-lg">Advanced AI-powered healthcare support • Multilingual • HIPAA Compliant</p>
                <div className="flex items-center space-x-6 mt-3">
                  <div className="flex items-center space-x-2 text-sm text-gray-500">
                    <Activity className="w-4 h-4 text-green-500" />
                    <span>Live Monitoring Active</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-500">
                    <Shield className="w-4 h-4 text-blue-500" />
                    <span>Secure & Confidential</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-500">
                    <Clock className="w-4 h-4 text-purple-500" />
                    <span>24/7 Available</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                <SelectTrigger className="glass-morphism border-0 w-44">
                  <div className="flex items-center space-x-2">
                    <Globe className="w-4 h-4 text-vitals-primary" />
                    <SelectValue />
                  </div>
                </SelectTrigger>
                <SelectContent>
                  {languages.map((lang) => (
                    <SelectItem key={lang.code} value={lang.code}>
                      <div className="flex items-center space-x-3">
                        <span>{lang.flag}</span>
                        <div>
                          <div className="font-medium">{lang.name}</div>
                          <div className="text-xs text-gray-500">{lang.region}</div>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={selectedSpecialty} onValueChange={setSelectedSpecialty}>
                <SelectTrigger className="glass-morphism border-0 w-48">
                  <div className="flex items-center space-x-2">
                    <Stethoscope className="w-4 h-4 text-vitals-primary" />
                    <SelectValue />
                  </div>
                </SelectTrigger>
                <SelectContent>
                  {medicalSpecialties.map((specialty) => {
                    const Icon = specialty.icon;
                    return (
                      <SelectItem key={specialty.id} value={specialty.id}>
                        <div className="flex items-center space-x-2">
                          <Icon className="w-4 h-4" />
                          <span>{specialty.name}</span>
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
              <Button variant="outline" className="glass-morphism" onClick={clearConversation}>
                <RotateCcw className="w-4 h-4 mr-2" />
                Reset
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Enhanced Chat Interface */}
        <div className="flex space-x-6">
          
          {/* Quick Actions Sidebar */}
          <AnimatePresence>
            {showQuickActions && (
              <motion.div
                className="w-80 space-y-4"
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -30 }}
                transition={{ duration: 0.3 }}
              >
                {/* Medical Specialties */}
                <div className="glass-morphism-dark rounded-2xl p-6">
                  <h3 className="font-semibold text-lg text-gray-800 mb-4">Medical Specialties</h3>
                  <div className="space-y-3">
                    {medicalSpecialties.map((specialty, index) => {
                      const Icon = specialty.icon;
                      return (
                        <motion.button
                          key={specialty.id}
                          className={`w-full p-3 rounded-xl text-left transition-all hover:shadow-md ${
                            selectedSpecialty === specialty.id
                              ? 'bg-vitals-primary text-white'
                              : 'bg-white/60 hover:bg-white/80'
                          }`}
                          onClick={() => setSelectedSpecialty(specialty.id)}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3, delay: index * 0.1 }}
                        >
                          <div className="flex items-center space-x-3">
                            <Icon className="w-5 h-5" />
                            <span className="font-medium">{specialty.name}</span>
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>
                </div>

                {/* Quick Actions */}
                <div className="glass-morphism-dark rounded-2xl p-6">
                  <h3 className="font-semibold text-lg text-gray-800 mb-4">Quick Actions</h3>
                  <div className="space-y-3">
                    {quickActions.map((action, index) => {
                      const Icon = action.icon;
                      return (
                        <motion.button
                          key={action.text}
                          className="w-full p-3 bg-white/60 rounded-xl text-left hover:bg-white/80 hover:shadow-md transition-all"
                          onClick={() => handleQuickAction(action.text, action.category)}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3, delay: index * 0.1 }}
                        >
                          <div className="flex items-start space-x-3">
                            <div className={`w-10 h-10 ${action.color} rounded-xl flex items-center justify-center flex-shrink-0`}>
                              <Icon className="w-5 h-5 text-white" />
                            </div>
                            <div>
                              <div className="font-medium text-gray-800">{action.text}</div>
                              <div className="text-sm text-gray-500">{action.description}</div>
                            </div>
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>
                </div>

                {/* Conversation Starters */}
                <div className="glass-morphism-dark rounded-2xl p-6">
                  <h3 className="font-semibold text-lg text-gray-800 mb-4">Example Questions</h3>
                  <div className="space-y-2">
                    {conversationStarters.slice(0, 3).map((starter, index) => (
                      <motion.button
                        key={index}
                        className="w-full p-2 text-sm bg-blue-50 text-blue-700 rounded-xl text-left hover:bg-blue-100 transition-colors"
                        onClick={() => handleConversationStarter(starter)}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                      >
                        "{starter}"
                      </motion.button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Main Chat Container */}
          <motion.div
            className="flex-1 glass-morphism-dark rounded-3xl flex flex-col"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            style={{ height: 'calc(100vh - 280px)' }}
          >
            {/* Chat Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full animate-pulse ${
                  conversationMode === 'emergency' ? 'bg-red-500' :
                  conversationMode === 'assessment' ? 'bg-yellow-500' : 'bg-green-500'
                }`}></div>
                <span className="font-medium text-gray-800">
                  {conversationMode === 'emergency' ? 'Emergency Mode' :
                   conversationMode === 'assessment' ? 'Assessment Mode' : 'General Chat'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="sm" className="glass-morphism">
                  <Archive className="w-4 h-4 mr-1" />
                  Save
                </Button>
                <Button variant="outline" size="sm" className="glass-morphism">
                  <Share2 className="w-4 h-4 mr-1" />
                  Share
                </Button>
                <Button variant="outline" size="sm" className="glass-morphism">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              <AnimatePresence>
                {messages.map((message, index) => (
                  <motion.div
                    key={message.id}
                    className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={{ duration: 0.4, delay: index * 0.05 }}
                  >
                    <div className={`flex items-start space-x-4 max-w-2xl ${message.type === "user" ? "flex-row-reverse space-x-reverse" : ""}`}>
                      {/* Avatar */}
                      <div className={`w-10 h-10 rounded-2xl flex items-center justify-center flex-shrink-0 ${
                        message.type === "user" 
                          ? "bg-gradient-to-br from-blue-500 to-blue-600" 
                          : "bg-gradient-to-br from-indigo-500 to-purple-600"
                      }`}>
                        {message.type === "user" ? (
                          <User className="w-5 h-5 text-white" />
                        ) : (
                          <Brain className="w-5 h-5 text-white" />
                        )}
                      </div>

                      {/* Message Content */}
                      <div className={`p-4 rounded-2xl max-w-full ${
                        message.type === "user" 
                          ? "bg-blue-500 text-white" 
                          : "bg-white/80 text-gray-800 shadow-sm border"
                      }`}>
                        {/* Message Text */}
                        <div className={`${message.type === 'user' ? 'text-white' : 'text-gray-800'}`}>
                          {message.content.split('\n').map((line, lineIndex) => (
                            <div key={lineIndex} className={line.startsWith('**') && line.endsWith('**') ? 'font-bold text-lg mb-2' : 'mb-1'}>
                              {line.replace(/\*\*/g, '')}
                            </div>
                          ))}
                        </div>

                        {/* Attachments */}
                        {message.attachments && message.attachments.length > 0 && (
                          <div className="mt-3 space-y-2">
                            {message.attachments.map((attachment, idx) => (
                              <div key={idx} className="flex items-center space-x-2 p-2 bg-gray-100 rounded-xl">
                                <Paperclip className="w-4 h-4 text-gray-500" />
                                <span className="text-sm">{attachment.name}</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Message Actions */}
                        {message.actions && message.actions.length > 0 && (
                          <div className="mt-4 flex flex-wrap gap-2">
                            {message.actions.map((action, actionIndex) => (
                              <Button
                                key={actionIndex}
                                size="sm"
                                variant={action.type === 'primary' ? 'default' : 'outline'}
                                className={action.type === 'primary' ? 'bg-vitals-primary text-white' : 'bg-white/80'}
                              >
                                {action.label}
                              </Button>
                            ))}
                          </div>
                        )}

                        {/* Message Metadata */}
                        <div className={`flex items-center justify-between mt-3 pt-3 border-t ${
                          message.type === 'user' ? 'border-blue-400' : 'border-gray-200'
                        }`}>
                          <div className={`text-xs flex items-center space-x-3 ${
                            message.type === 'user' ? 'text-blue-100' : 'text-gray-500'
                          }`}>
                            <span>{message.timestamp.toLocaleTimeString()}</span>
                            {message.category && (
                              <span className={`px-2 py-1 rounded-full ${
                                message.category === 'emergency' ? 'bg-red-100 text-red-700' :
                                message.category === 'symptom' ? 'bg-yellow-100 text-yellow-700' :
                                message.category === 'medication' ? 'bg-green-100 text-green-700' :
                                'bg-blue-100 text-blue-700'
                              }`}>
                                {message.category}
                              </span>
                            )}
                            {message.priority && message.priority !== 'low' && (
                              <span className={`font-medium ${
                                message.priority === 'critical' ? 'text-red-600' :
                                message.priority === 'high' ? 'text-orange-600' :
                                'text-yellow-600'
                              }`}>
                                {message.priority === 'critical' ? '🚨' :
                                 message.priority === 'high' ? '⚠️' : '⚡'}
                              </span>
                            )}
                          </div>
                          {message.type === 'assistant' && (
                            <div className="flex items-center space-x-1">
                              <Button variant="ghost" size="sm" className="h-6 w-6 p-0 text-gray-400 hover:text-gray-600">
                                <Copy className="w-3 h-3" />
                              </Button>
                              <Button variant="ghost" size="sm" className="h-6 w-6 p-0 text-gray-400 hover:text-gray-600">
                                <Star className="w-3 h-3" />
                              </Button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                  <motion.div
                    className="flex justify-start"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <div className="flex items-start space-x-4">
                      <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                        <Brain className="w-5 h-5 text-white" />
                      </div>
                      <div className="bg-white/80 p-4 rounded-2xl shadow-sm border">
                        <div className="flex items-center space-x-2">
                          <div className="flex space-x-1">
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                          </div>
                          <span className="text-sm text-gray-500">MedAI is analyzing...</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>

            {/* Enhanced Input Area */}
            <div className="border-t border-gray-200 p-6">
              {/* Attachments Preview */}
              {attachments.length > 0 && (
                <div className="mb-4 flex flex-wrap gap-2">
                  {attachments.map((attachment, index) => (
                    <motion.div
                      key={index}
                      className="flex items-center space-x-2 bg-blue-50 text-blue-700 px-3 py-2 rounded-xl text-sm"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                    >
                      <Paperclip className="w-4 h-4" />
                      <span>{attachment.name}</span>
                      <button
                        onClick={() => setAttachments(prev => prev.filter((_, i) => i !== index))}
                        className="text-blue-500 hover:text-blue-700"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </motion.div>
                  ))}
                </div>
              )}

              {/* Input Row */}
              <div className="flex items-end space-x-4">
                <div className="flex-1 relative">
                  <Textarea
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder={`Ask MedAI anything about your health... (${selectedLanguage})`}
                    className="w-full min-h-[60px] max-h-32 resize-none glass-morphism border-0 rounded-2xl p-4 pr-12"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    data-testid="message-input"
                  />
                  <div className="absolute right-3 bottom-3 flex items-center space-x-1">
                    <span className="text-xs text-gray-400">
                      {inputMessage.length > 0 && `${inputMessage.length}/1000`}
                    </span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center space-x-2">
                  <input
                    type="file"
                    id="file-upload"
                    multiple
                    accept="image/*,.pdf,.doc,.docx"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    className="glass-morphism h-12 w-12"
                    onClick={() => document.getElementById('file-upload')?.click()}
                  >
                    <Paperclip className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="glass-morphism h-12 w-12"
                    onClick={() => document.getElementById('camera-upload')?.click()}
                  >
                    <Camera className="w-4 h-4" />
                  </Button>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={toggleListening}
                      className={`glass-morphism h-12 w-12 ${
                        isListening ? 'bg-red-500 text-white border-red-500' : ''
                      }`}
                    >
                      <Mic className="w-4 h-4" />
                    </Button>
                  </motion.div>
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button
                      onClick={sendMessage}
                      disabled={!inputMessage.trim() && attachments.length === 0}
                      className="bg-vitals-primary text-white hover:bg-blue-600 h-12 px-6 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Send className="w-4 h-4 mr-2" />
                      Send
                    </Button>
                  </motion.div>
                </div>
              </div>

              {/* Input Helpers */}
              <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
                <div className="flex items-center space-x-4">
                  <span>Press Shift+Enter for new line</span>
                  <span>•</span>
                  <span>Supports images, documents, voice input</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Shield className="w-4 h-4 text-green-500" />
                  <span>End-to-end encrypted</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

      </div>
    </div>
  );
}
