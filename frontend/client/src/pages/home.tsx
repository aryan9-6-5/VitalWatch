import { motion } from "framer-motion";
import { Link } from "wouter";
import { Play, ShieldCheck, Users, Star, Activity, Bell, Monitor, Brain, Pill, PhoneCall, Check, ArrowRight, 
         BarChart3, TrendingUp, Shield, Zap, Clock, Globe, Award, Heart, Smartphone, 
         Cloud, Lock, Wifi, MessageCircle, Stethoscope, FileText, Calendar, Target, 
         AlertTriangle, Bot, Wind } from "lucide-react";
import { Button } from "@/components/ui/button";
import FloatingDashboard from "@/components/FloatingDashboard";

export default function Home() {
  const features = [
    {
      icon: Activity,
      title: "🩺 Real-Time Vitals Monitoring",
      description: "Monitor HR, BP, SpO₂, Temperature, RR, and CO for multiple family members. Connects with iHealth, Dozee, Omron, and Wellue devices.",
      link: "/vitals",
      gradient: "from-vitals-healthy to-green-600"
    },
    {
      icon: Bell,
      title: "🚨 Smart Alerts + Escalation",
      description: "Classify abnormal readings, notify family and clinic instantly. Emergency triggers with WhatsApp, SMS, and in-app notifications.",
      link: "/alerts",
      gradient: "from-vitals-warning to-orange-500"
    },
    {
      icon: Monitor,
      title: "📊 Dual Dashboards",
      description: "Medical-grade doctor dashboard for professionals and empathetic family dashboard for caregivers. Tailored experiences for every user.",
      link: "/doctor-dashboard",
      gradient: "from-vitals-primary to-blue-600"
    },
    {
      icon: Brain,
      title: "🧠 AI Health Assistant",
      description: "Multilingual support in Telugu, Hindi, English. Voice/text symptom analysis, medicine suggestions, and critical sign escalation.",
      link: "/ai-assistant",
      gradient: "from-purple-500 to-purple-600"
    },
    {
      icon: Pill,
      title: "📦 ePharmacy Shop",
      description: "Browse medications with filters, add to cart, and checkout seamlessly. Integrated with 1mg and PharmEasy for reliable delivery.",
      link: "/pharmacy",
      gradient: "from-green-500 to-emerald-600"
    },
    {
      icon: PhoneCall,
      title: "🆘 Emergency System",
      description: "1-click GPS + health snapshot to family. 3-tap alerts ambulance, clinic, and all caretakers. Always accessible SOS button.",
      link: "/emergency",
      gradient: "from-vitals-critical to-red-600"
    }
  ];

  const testimonials = [
    {
      name: "Priya Sharma",
      location: "Bangalore, Karnataka",
      quote: "VitalsBridge helped me save my father's life. The alert came at 2 AM when his blood pressure spiked, and we rushed to the hospital in time.",
      avatar: "https://images.unsplash.com/photo-1582750433449-648ed127bb54?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=300"
    },
    {
      name: "Dr. Rajesh Kumar",
      location: "Mumbai, Maharashtra",
      quote: "As a doctor, I appreciate the medical-grade accuracy. It's like having a 24/7 monitoring system for all my patients.",
      avatar: "https://images.unsplash.com/photo-1607990281513-2c110a25bd8c?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=300"
    },
    {
      name: "Anita Gupta",
      location: "Delhi, NCR",
      quote: "Managing my elderly parents' health from Delhi was impossible until VitalsBridge. Now I sleep peacefully knowing they're monitored.",
      avatar: "https://images.unsplash.com/photo-1494790108755-2616b612b786?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=300"
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center hero-gradient overflow-hidden">
        {/* Floating Background Elements */}
        <div className="absolute inset-0">
          <motion.div
            className="absolute top-20 left-10 w-20 h-20 bg-vitals-primary/10 rounded-full"
            animate={{ y: [0, -20, 0] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            className="absolute top-40 right-20 w-16 h-16 bg-vitals-healthy/10 rounded-full"
            animate={{ y: [0, -20, 0] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 1 }}
          />
          <motion.div
            className="absolute bottom-40 left-1/4 w-12 h-12 bg-vitals-warning/10 rounded-full"
            animate={{ y: [0, -20, 0] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 2 }}
          />
          <motion.div
            className="absolute bottom-20 right-1/3 w-24 h-24 bg-vitals-primary/5 rounded-full"
            animate={{ y: [0, -20, 0] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
          />
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <motion.div
              className="text-center lg:text-left space-y-8"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <div className="space-y-4">
                <h1 className="font-poppins font-bold text-5xl lg:text-6xl text-gray-800 leading-tight">
                  Peace of Health for Every{" "}
                  <span className="text-vitals-primary relative">
                    Heartbeat
                    <motion.div
                      className="absolute -top-2 -right-2 w-4 h-4 bg-vitals-critical rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    />
                  </span>
                  {" "}You Care About
                </h1>
                <p className="text-xl text-gray-600 font-inter leading-relaxed">
                  Track real-time vitals, alerts, medications, consultations, and insurance — anywhere.
                  Making preventive care visible, accessible, and human.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <Link href="/family-dashboard" data-testid="button-start-trial">
                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                    <Button className="bg-vitals-primary text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-blue-600 transition-all shadow-lg hover:shadow-xl">
                      Start Free Trial
                    </Button>
                  </motion.div>
                </Link>
                <Button
                  variant="ghost"
                  className="glass-morphism text-vitals-primary px-8 py-4 rounded-xl font-semibold text-lg hover:bg-blue-50 transition-all shadow-lg"
                  data-testid="button-watch-demo"
                >
                  <Play className="w-5 h-5 inline mr-2" />
                  Watch How It Works
                </Button>
              </div>

              {/* Trust Indicators */}
              <div className="flex items-center justify-center lg:justify-start space-x-6 text-sm text-gray-500">
                <div className="flex items-center space-x-2">
                  <ShieldCheck className="w-4 h-4 text-vitals-healthy" />
                  <span>HIPAA Compliant</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Users className="w-4 h-4 text-vitals-primary" />
                  <span>10,000+ Families</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Star className="w-4 h-4 text-vitals-warning" />
                  <span>4.9/5 Rating</span>
                </div>
              </div>
            </motion.div>

            {/* Right Content - Floating Dashboard */}
            <FloatingDashboard />
          </div>
        </div>
      </section>

      {/* Live Statistics Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-12"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-4xl text-gray-800 mb-4">Trusted by Families Across India</h2>
            <p className="text-xl text-gray-600">Real-time health monitoring that saves lives every day</p>
          </motion.div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { number: "50,000+", label: "Lives Protected", icon: Shield, color: "text-green-500" },
              { number: "99.2%", label: "Uptime Reliability", icon: Zap, color: "text-blue-500" },
              { number: "2.5 min", label: "Avg Response Time", icon: Clock, color: "text-purple-500" },
              { number: "15+", label: "Device Integrations", icon: Smartphone, color: "text-orange-500" }
            ].map((stat, index) => {
              const Icon = stat.icon;
              return (
                <motion.div
                  key={stat.label}
                  className="text-center"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                >
                  <div className="relative mb-4">
                    <Icon className={`w-12 h-12 mx-auto ${stat.color}`} />
                    <div className="absolute -top-2 -right-2 w-6 h-6 bg-vitals-primary rounded-full flex items-center justify-center">
                      <TrendingUp className="w-3 h-3 text-white" />
                    </div>
                  </div>
                  <motion.div
                    className="text-3xl font-bold text-gray-800 mb-2"
                    initial={{ scale: 0 }}
                    whileInView={{ scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
                  >
                    {stat.number}
                  </motion.div>
                  <p className="text-gray-600 font-medium">{stat.label}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Interactive Demo Section */}
      <section className="py-20 bg-gradient-to-br from-vitals-primary/5 to-blue-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="font-poppins font-bold text-4xl text-gray-800 mb-6">
                See VitalsBridge in Action
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Watch how our AI-powered health assistant monitors vital signs, 
                provides instant alerts, and coordinates emergency care in real-time.
              </p>
              
              <div className="space-y-4 mb-8">
                {[
                  { icon: Activity, text: "Real-time vital signs monitoring with smart alerts" },
                  { icon: Brain, text: "AI-powered health analysis in multiple languages" },
                  { icon: MessageCircle, text: "Instant family and doctor notifications" },
                  { icon: Shield, text: "24/7 emergency response coordination" }
                ].map((item, index) => {
                  const Icon = item.icon;
                  return (
                    <motion.div
                      key={index}
                      className="flex items-center space-x-3"
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                    >
                      <div className="w-8 h-8 bg-vitals-primary rounded-lg flex items-center justify-center">
                        <Icon className="w-4 h-4 text-white" />
                      </div>
                      <span className="text-gray-700">{item.text}</span>
                    </motion.div>
                  );
                })}
              </div>
              
              <Button className="bg-vitals-primary text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-blue-600 transition-all shadow-lg">
                <Play className="w-5 h-5 mr-2" />
                Watch 3-Min Demo
              </Button>
            </motion.div>
            
            <motion.div
              className="relative"
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <div className="bg-white rounded-3xl p-8 shadow-2xl">
                <div className="flex items-center space-x-3 mb-6">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-gray-500 ml-4">VitalsBridge Demo</span>
                </div>
                
                <div className="space-y-4">
                  <div className="p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      <span className="font-semibold text-red-700">Critical Alert</span>
                    </div>
                    <p className="text-red-600 text-sm">Mr. Rajesh - BP: 180/110 • Heart Rate: 120</p>
                    <p className="text-xs text-red-500 mt-1">Emergency contacts notified • Ambulance dispatched</p>
                  </div>
                  
                  <div className="p-4 bg-green-50 border-l-4 border-green-500 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Heart className="w-5 h-5 text-green-500" />
                      <span className="font-semibold text-green-700">Normal Status</span>
                    </div>
                    <p className="text-green-600 text-sm">Mrs. Priya - All vitals within range</p>
                    <p className="text-xs text-green-500 mt-1">Medication reminder sent • Next checkup scheduled</p>
                  </div>
                  
                  <div className="p-4 bg-blue-50 border-l-4 border-blue-500 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Bot className="w-5 h-5 text-blue-500" />
                      <span className="font-semibold text-blue-700">AI Assistant</span>
                    </div>
                    <p className="text-blue-600 text-sm">"Based on your symptoms, I recommend scheduling a consultation within 24 hours."</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Health Conditions Coverage */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-4xl text-gray-800 mb-4">
              Comprehensive Health Condition Support
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Specialized monitoring and care protocols for the most common health conditions affecting Indian families
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { condition: "Hypertension", patients: "2.5M+", icon: Heart, color: "from-red-400 to-red-600", description: "Blood pressure monitoring with instant alerts for dangerous spikes" },
              { condition: "Diabetes", patients: "1.8M+", icon: Target, color: "from-blue-400 to-blue-600", description: "Glucose tracking, medication reminders, and dietary guidance" },
              { condition: "Heart Disease", patients: "950K+", icon: Activity, color: "from-purple-400 to-purple-600", description: "Cardiac rhythm monitoring and emergency response protocols" },
              { condition: "Respiratory Issues", patients: "720K+", icon: Wind, color: "from-green-400 to-green-600", description: "Oxygen saturation tracking and breathing pattern analysis" },
              { condition: "Elderly Care", patients: "3.2M+", icon: Shield, color: "from-orange-400 to-orange-600", description: "Comprehensive monitoring for age-related health concerns" },
              { condition: "Mental Health", patients: "1.1M+", icon: Brain, color: "from-indigo-400 to-indigo-600", description: "Stress monitoring and emotional wellbeing support" }
            ].map((item, index) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={item.condition}
                  className="glass-morphism-dark rounded-2xl p-6 hover:shadow-xl transition-all"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                  whileHover={{ y: -5 }}
                >
                  <div className={`w-16 h-16 bg-gradient-to-br ${item.color} rounded-2xl flex items-center justify-center mb-4`}>
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="font-poppins font-semibold text-xl text-gray-800 mb-2">{item.condition}</h3>
                  <p className="text-vitals-primary font-semibold text-lg mb-3">{item.patients} patients monitored</p>
                  <p className="text-gray-600 text-sm">{item.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Technology Integration Section */}
      <section className="py-20 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-4xl text-gray-800 mb-4">
              Advanced Technology Stack
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Built with enterprise-grade security and reliability for healthcare data
            </p>
          </motion.div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-8">
            {[
              { name: "AI/ML Engine", icon: Brain },
              { name: "Cloud Infrastructure", icon: Cloud },
              { name: "End-to-End Encryption", icon: Lock },
              { name: "Real-time Sync", icon: Wifi },
              { name: "Mobile First", icon: Smartphone },
              { name: "HIPAA Compliant", icon: ShieldCheck }
            ].map((tech, index) => {
              const Icon = tech.icon;
              return (
                <motion.div
                  key={tech.name}
                  className="text-center"
                  initial={{ opacity: 0, scale: 0.5 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                >
                  <div className="w-16 h-16 mx-auto bg-vitals-primary/10 rounded-2xl flex items-center justify-center mb-3 hover:bg-vitals-primary/20 transition-colors">
                    <Icon className="w-8 h-8 text-vitals-primary" />
                  </div>
                  <p className="text-sm font-medium text-gray-700">{tech.name}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Core Features Section */}
      <section className="py-20 bg-gradient-to-b from-white to-vitals-card-bg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-4xl text-gray-800 mb-4">Complete Health Ecosystem</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Everything you need to monitor, manage, and maintain your family's health in one unified platform
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
              >
                <Link href={feature.link} data-testid={`feature-${feature.title.toLowerCase().replace(/[^a-z0-9]/g, '')}`}>
                  <motion.div
                    className={`glass-morphism-dark rounded-3xl p-8 hover:shadow-xl transition-all cursor-pointer h-full ${
                      feature.title.includes("Emergency") ? "emergency-glow" : ""
                    }`}
                    whileHover={{ scale: 1.05, y: -5 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className={`w-16 h-16 bg-gradient-to-br ${feature.gradient} rounded-2xl flex items-center justify-center mb-6`}>
                      <feature.icon className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="font-poppins font-semibold text-xl text-gray-800 mb-4">{feature.title}</h3>
                    <p className="text-gray-600 mb-6">{feature.description}</p>
                    <div className="flex items-center text-vitals-primary font-medium hover:text-blue-600">
                      <span>Learn More</span>
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </div>
                  </motion.div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Security & Compliance Section */}
      <section className="py-20 bg-gradient-to-br from-gray-800 to-gray-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-4xl text-white mb-4">
              Enterprise-Grade Security & Compliance
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Your family's health data is protected by the highest security standards in healthcare technology
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Lock,
                title: "End-to-End Encryption",
                description: "AES-256 encryption ensures your health data is always secure, both in transit and at rest."
              },
              {
                icon: ShieldCheck,
                title: "HIPAA Compliance",
                description: "Fully compliant with healthcare privacy regulations, audited by third-party security firms."
              },
              {
                icon: Award,
                title: "ISO 27001 Certified",
                description: "International security management standards for handling sensitive healthcare information."
              }
            ].map((item, index) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={item.title}
                  className="text-center p-8"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.8, delay: index * 0.2 }}
                >
                  <div className="w-20 h-20 mx-auto bg-white/10 rounded-2xl flex items-center justify-center mb-6">
                    <Icon className="w-10 h-10 text-vitals-primary" />
                  </div>
                  <h3 className="font-semibold text-xl text-white mb-4">{item.title}</h3>
                  <p className="text-gray-300">{item.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-4xl text-gray-800 mb-4">Trusted by Families Across India</h2>
            <p className="text-xl text-gray-600">Real stories from families who found peace of mind with VitalsBridge</p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={testimonial.name}
                className="glass-morphism-dark rounded-3xl p-8"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                whileHover={{ y: -5 }}
              >
                <img
                  src={testimonial.avatar}
                  alt={testimonial.name}
                  className="w-20 h-20 rounded-full mx-auto mb-6 object-cover"
                />
                <blockquote className="text-gray-600 mb-6 italic">"{testimonial.quote}"</blockquote>
                <div className="text-center">
                  <div className="font-semibold text-gray-800">{testimonial.name}</div>
                  <div className="text-gray-500">{testimonial.location}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final Call-to-Action Section */}
      <section className="py-24 bg-gradient-to-br from-vitals-primary to-blue-700 text-white relative overflow-hidden">
        {/* Background Animation */}
        <div className="absolute inset-0">
          <motion.div
            className="absolute top-10 left-10 w-32 h-32 bg-white/5 rounded-full"
            animate={{ y: [0, -30, 0], rotate: [0, 180, 360] }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          />
          <motion.div
            className="absolute bottom-20 right-20 w-24 h-24 bg-white/10 rounded-full"
            animate={{ y: [0, -20, 0], rotate: [360, 180, 0] }}
            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          />
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.div
            className="text-center"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-5xl mb-6">
              Ready to Transform Your Family's Health Journey?
            </h2>
            <p className="text-xl text-blue-100 mb-12 max-w-3xl mx-auto">
              Join 50,000+ families who trust VitalsBridge to monitor, protect, and care for their loved ones. 
              Start your free trial today and experience peace of mind like never before.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-12">
              <Link href="/family-dashboard">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button className="bg-white text-vitals-primary px-12 py-4 rounded-2xl font-bold text-lg hover:bg-gray-100 transition-all shadow-2xl">
                    <Activity className="w-6 h-6 mr-3" />
                    Start Free Trial Now
                  </Button>
                </motion.div>
              </Link>
              
              <Link href="/ai-assistant">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button 
                    variant="outline" 
                    className="border-2 border-white/30 text-white px-8 py-4 rounded-2xl font-semibold text-lg hover:bg-white/10 transition-all backdrop-blur-sm"
                  >
                    <Bot className="w-6 h-6 mr-3" />
                    Try AI Assistant
                  </Button>
                </motion.div>
              </Link>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
              {[
                { icon: Clock, text: "Setup in 5 minutes", subtext: "Quick & easy onboarding" },
                { icon: Shield, text: "No commitment", subtext: "Cancel anytime" },
                { icon: Heart, text: "24/7 support", subtext: "Always here to help" }
              ].map((item, index) => {
                const Icon = item.icon;
                return (
                  <motion.div
                    key={index}
                    className="flex flex-col items-center"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  >
                    <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center mb-4">
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <h4 className="font-semibold text-lg mb-2">{item.text}</h4>
                    <p className="text-blue-100 text-sm">{item.subtext}</p>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Pricing Preview Section */}
      <section className="py-20 bg-gradient-to-b from-vitals-card-bg to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="font-poppins font-bold text-4xl text-gray-800 mb-4">Choose Your Health Plan</h2>
            <p className="text-xl text-gray-600 mb-8">Transparent pricing for every family size</p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* Basic Plan */}
            <motion.div
              className="glass-morphism-dark rounded-3xl p-8 border-2 border-transparent hover:border-vitals-primary/20 transition-all"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="text-center mb-8">
                <h3 className="font-poppins font-bold text-2xl text-gray-800 mb-2">Basic Plan</h3>
                <div className="text-4xl font-bold text-vitals-primary mb-2">
                  ₹400<span className="text-lg text-gray-500">/month</span>
                </div>
                <p className="text-gray-600">Perfect for small families</p>
              </div>

              <div className="space-y-4 mb-8">
                {[
                  "Up to 3 family members",
                  "Real-time vitals monitoring",
                  "Smart alerts & notifications",
                  "AI Health Assistant",
                  "Basic emergency support"
                ].map((feature) => (
                  <div key={feature} className="flex items-center space-x-3">
                    <Check className="w-5 h-5 text-vitals-healthy" />
                    <span>{feature}</span>
                  </div>
                ))}
              </div>

              <Link href="/pricing" data-testid="button-start-basic">
                <Button className="w-full bg-vitals-primary text-white py-3 rounded-xl font-semibold hover:bg-blue-600 transition-colors">
                  Start Basic Plan
                </Button>
              </Link>
            </motion.div>

            {/* Pro Plan */}
            <motion.div
              className="glass-morphism-dark rounded-3xl p-8 border-2 border-vitals-primary shadow-xl relative"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                <span className="bg-vitals-primary text-white px-6 py-2 rounded-full text-sm font-semibold">
                  Most Popular
                </span>
              </div>

              <div className="text-center mb-8">
                <h3 className="font-poppins font-bold text-2xl text-gray-800 mb-2">Pro Plan</h3>
                <div className="text-4xl font-bold text-vitals-primary mb-2">
                  ₹700<span className="text-lg text-gray-500">/month</span>
                </div>
                <p className="text-gray-600">For larger families & priority care</p>
              </div>

              <div className="space-y-4 mb-8">
                {[
                  "Up to 6 family members",
                  "All Basic features",
                  "Priority doctor consultations",
                  "Advanced health trends",
                  "24/7 emergency response",
                  "Insurance document storage"
                ].map((feature) => (
                  <div key={feature} className="flex items-center space-x-3">
                    <Check className="w-5 h-5 text-vitals-healthy" />
                    <span>{feature}</span>
                  </div>
                ))}
              </div>

              <Link href="/pricing" data-testid="button-start-pro">
                <Button className="w-full bg-vitals-primary text-white py-3 rounded-xl font-semibold hover:bg-blue-600 transition-colors">
                  Start Pro Plan
                </Button>
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            {/* Brand */}
            <div className="col-span-2">
              <div className="flex items-center space-x-2 mb-6">
                <div className="w-8 h-8 bg-vitals-primary rounded-xl flex items-center justify-center">
                  <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
                    <div className="w-5 h-5 text-white">❤️</div>
                  </motion.div>
                </div>
                <span className="font-poppins font-bold text-xl">VitalsBridge</span>
              </div>
              <p className="text-gray-400 mb-6 max-w-md">
                Peace of Health for Every Heartbeat You Care About. Making preventive care visible, accessible, and human for Indian families.
              </p>
            </div>

            {/* Product */}
            <div>
              <h3 className="font-semibold text-lg mb-4">Product</h3>
              <div className="space-y-3">
                <Link href="/family-dashboard">
                  <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">Family Dashboard</div>
                </Link>
                <Link href="/doctor-dashboard">
                  <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">Doctor Portal</div>
                </Link>
                <Link href="/pharmacy">
                  <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">ePharmacy</div>
                </Link>
                <Link href="/emergency">
                  <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">Emergency</div>
                </Link>
              </div>
            </div>

            {/* Support */}
            <div>
              <h3 className="font-semibold text-lg mb-4">Support</h3>
              <div className="space-y-3">
                <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">Help Center</div>
                <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">Privacy Policy</div>
                <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">Terms of Service</div>
                <div className="text-gray-400 hover:text-white transition-colors cursor-pointer">Contact Us</div>
              </div>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-12 pt-8 text-center text-gray-400">
            <p>&copy; 2024 VitalsBridge. All rights reserved. Made with ❤️ for Indian families.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
