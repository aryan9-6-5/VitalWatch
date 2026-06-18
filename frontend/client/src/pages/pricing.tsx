import { motion } from "framer-motion";
import { useState } from "react";
import { Check, Home, Users, Building } from "lucide-react";
import { Button } from "@/components/ui/button";
import PricingCard from "@/components/PricingCard";

const plans = [
  {
    id: "basic",
    name: "Basic Plan",
    description: "Perfect for small families",
    icon: Home,
    monthlyPrice: 400,
    yearlyPrice: 320,
    color: "from-blue-400 to-blue-600",
    features: [
      "Up to 3 family members",
      "Real-time vitals monitoring",
      "Smart alerts & notifications",
      "AI Health Assistant",
      "Basic emergency support",
      "ePharmacy access",
      "Health trends (basic)"
    ],
    ctaText: "Start Basic Plan",
    note: "7-day free trial • No setup fees"
  },
  {
    id: "pro",
    name: "Pro Plan",
    description: "For larger families & priority care",
    icon: Users,
    monthlyPrice: 700,
    yearlyPrice: 560,
    color: "from-vitals-primary to-blue-600",
    popular: true,
    features: [
      "Up to 6 family members",
      "All Basic features included",
      "Priority doctor consultations",
      "Advanced health trends & analytics",
      "24/7 emergency response",
      "Insurance document storage",
      "Device rental discounts",
      "Family health reports"
    ],
    ctaText: "Start Pro Plan",
    note: "14-day free trial • Cancel anytime"
  },
  {
    id: "enterprise",
    name: "Enterprise",
    description: "For clinics & large families",
    icon: Building,
    monthlyPrice: null,
    yearlyPrice: null,
    color: "from-purple-400 to-purple-600",
    features: [
      "Unlimited family members",
      "All Pro features included",
      "Custom integrations",
      "Dedicated support manager",
      "White-label options",
      "API access",
      "Advanced analytics dashboard",
      "SLA guarantees"
    ],
    ctaText: "Contact Sales",
    ctaVariant: "purple",
    note: "Custom demo • Volume discounts"
  }
];

const faqs = [
  {
    question: "Can I change plans anytime?",
    answer: "Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately, and we'll prorate the billing accordingly."
  },
  {
    question: "What devices are supported?",
    answer: "We support devices from iHealth, Dozee, Omron, Wellue, and other major brands. Full compatibility list available in our device marketplace."
  },
  {
    question: "Is my health data secure?",
    answer: "Absolutely. We're HIPAA compliant with end-to-end encryption, secure cloud storage, and regular security audits to protect your family's health information."
  },
  {
    question: "Do I need separate devices for each family member?",
    answer: "Not necessarily. Many devices can be shared, and our app helps you track which family member's readings are being recorded."
  },
  {
    question: "What's included in emergency response?",
    answer: "Emergency response includes instant family notifications, GPS location sharing, health snapshot delivery, and coordination with local emergency services when needed."
  },
  {
    question: "Can doctors outside India use this platform?",
    answer: "Currently, VitalsBridge is optimized for the Indian healthcare system. We're exploring expansion to other markets - contact us for updates."
  }
];

export default function Pricing() {
  const [isYearly, setIsYearly] = useState(false);

  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-vitals-card-bg to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="font-poppins font-bold text-4xl text-gray-800 mb-4">Choose Your Health Plan</h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Transparent pricing designed for Indian families. No hidden fees, cancel anytime.
          </p>

          {/* Billing Toggle */}
          <div className="inline-flex items-center bg-gray-100 rounded-xl p-1 mb-8">
            <motion.button
              className={`px-6 py-3 rounded-lg font-medium transition-all ${
                !isYearly ? "bg-white text-vitals-primary shadow-sm" : "text-gray-600"
              }`}
              onClick={() => setIsYearly(false)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              data-testid="monthly-toggle"
            >
              Monthly
            </motion.button>
            <motion.button
              className={`px-6 py-3 rounded-lg font-medium transition-all ${
                isYearly ? "bg-white text-vitals-primary shadow-sm" : "text-gray-600"
              }`}
              onClick={() => setIsYearly(true)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              data-testid="yearly-toggle"
            >
              Yearly
              <span className="bg-green-100 text-green-600 px-2 py-1 rounded-md text-xs ml-2">Save 20%</span>
            </motion.button>
          </div>
        </motion.div>

        {/* Pricing Plans */}
        <div className="grid lg:grid-cols-3 gap-8 mb-16">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.id}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: index * 0.2 }}
              className={plan.popular ? "transform scale-105" : ""}
            >
              <PricingCard plan={plan} isYearly={isYearly} />
            </motion.div>
          ))}
        </div>

        {/* FAQ Section */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          <h3 className="font-poppins font-bold text-2xl text-gray-800 mb-8 text-center">
            Frequently Asked Questions
          </h3>

          <div className="grid md:grid-cols-2 gap-8">
            {faqs.map((faq, index) => (
              <motion.div
                key={faq.question}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 1 + index * 0.1 }}
              >
                <h4 className="font-semibold text-gray-800 mb-2">{faq.question}</h4>
                <p className="text-gray-600 text-sm">{faq.answer}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
