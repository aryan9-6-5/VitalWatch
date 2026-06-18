import { Link, useLocation } from "wouter";
import { HeartPulse, Menu, Phone, Bot, MessageCircle, Pill, Activity, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { motion } from "framer-motion";

export default function Navigation() {
  const [location] = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { path: "/", label: "Home" },
    { path: "/doctor-dashboard", label: "Doctor Portal" },
    { path: "/family-dashboard", label: "Family Dashboard" },
    { path: "/vitals", label: "Vitals" },
    { path: "/pharmacy", label: "Pharmacy" },
    { path: "/emergency", label: "🆘 Emergency", className: "text-vitals-critical hover:text-red-600" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-morphism border-b border-blue-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" data-testid="logo-link">
            <motion.div 
              className="flex items-center space-x-2 cursor-pointer"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="w-8 h-8 bg-vitals-primary rounded-xl flex items-center justify-center">
                <HeartPulse className="w-5 h-5 text-white animate-heartbeat" />
              </div>
              <span className="font-poppins font-bold text-xl text-gray-800">VitalsBridge</span>
            </motion.div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6">
            {navItems.map((item) => (
              <Link key={item.path} href={item.path} data-testid={`nav-${item.label.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}`}>
                <motion.button
                  className={`text-gray-600 hover:text-vitals-primary transition-colors font-medium ${
                    location === item.path ? "text-vitals-primary" : ""
                  } ${item.className || ""}`}
                  whileHover={{ y: -2 }}
                  whileTap={{ y: 0 }}
                >
                  {item.label}
                </motion.button>
              </Link>
            ))}
            
            {/* AI Assistant Button */}
            <Link href="/ai-assistant" data-testid="nav-ai-assistant">
              <motion.div 
                whileHover={{ scale: 1.05 }} 
                whileTap={{ scale: 0.95 }}
                className="relative"
              >
                <Button 
                  variant="outline" 
                  className="glass-morphism border-vitals-primary/20 text-vitals-primary hover:bg-vitals-primary hover:text-white transition-all font-medium px-4 py-2 rounded-xl flex items-center space-x-2"
                >
                  <Bot className="w-4 h-4" />
                  <span className="hidden lg:inline">AI Assistant</span>
                </Button>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </motion.div>
            </Link>
            
            <Link href="/pricing" data-testid="nav-starttrial">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Button className="bg-vitals-primary text-white hover:bg-blue-600 transition-colors font-medium px-6 py-2 rounded-xl">
                  Start Free Trial
                </Button>
              </motion.div>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <Button
              variant="ghost"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              data-testid="mobile-menu-button"
            >
              <Menu className="w-6 h-6 text-gray-600" />
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="md:hidden border-t border-gray-200 mt-2 pt-4 pb-4"
          >
            <div className="flex flex-col space-y-3">
              {navItems.map((item) => (
                <Link key={item.path} href={item.path} data-testid={`mobile-nav-${item.label.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}`}>
                  <button
                    className={`text-left w-full text-gray-600 hover:text-vitals-primary transition-colors font-medium ${
                      location === item.path ? "text-vitals-primary" : ""
                    } ${item.className || ""}`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    {item.label}
                  </button>
                </Link>
              ))}
              
              <Link href="/ai-assistant" data-testid="mobile-nav-ai-assistant">
                <button
                  className="text-left w-full text-vitals-primary hover:text-blue-600 transition-colors font-medium flex items-center space-x-2"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <Bot className="w-4 h-4" />
                  <span>AI Assistant</span>
                </button>
              </Link>
              
              <Link href="/pricing" data-testid="mobile-nav-starttrial">
                <Button 
                  className="w-full bg-vitals-primary text-white hover:bg-blue-600 transition-colors font-medium"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Start Free Trial
                </Button>
              </Link>
            </div>
          </motion.div>
        )}
      </div>
    </nav>
  );
}
