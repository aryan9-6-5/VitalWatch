import { motion } from "framer-motion";
import { Check, LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PricingPlan {
  id: string;
  name: string;
  description: string;
  icon: LucideIcon;
  monthlyPrice: number | null;
  yearlyPrice: number | null;
  color: string;
  popular?: boolean;
  features: string[];
  ctaText: string;
  ctaVariant?: string;
  note: string;
}

interface PricingCardProps {
  plan: PricingPlan;
  isYearly: boolean;
}

export default function PricingCard({ plan, isYearly }: PricingCardProps) {
  const currentPrice = isYearly ? plan.yearlyPrice : plan.monthlyPrice;
  const originalPrice = isYearly ? plan.monthlyPrice : null;
  const savings = plan.monthlyPrice && plan.yearlyPrice 
    ? ((plan.monthlyPrice - plan.yearlyPrice) / plan.monthlyPrice * 100).toFixed(0)
    : null;

  const getButtonColor = () => {
    if (plan.ctaVariant === "purple") return "bg-purple-600 hover:bg-purple-700";
    return "bg-vitals-primary hover:bg-blue-600";
  };

  return (
    <motion.div
      className={`glass-morphism-dark rounded-3xl p-8 border-2 transition-all h-full flex flex-col ${
        plan.popular 
          ? "border-vitals-primary shadow-xl" 
          : "border-transparent hover:border-vitals-primary/20"
      }`}
      whileHover={{ y: -5, scale: 1.02 }}
      data-testid={`pricing-card-${plan.id}`}
    >
      {plan.popular && (
        <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
          <motion.span
            className="bg-vitals-primary text-white px-6 py-2 rounded-full text-sm font-semibold"
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            Most Popular
          </motion.span>
        </div>
      )}

      <div className="text-center mb-8">
        <div className={`w-16 h-16 bg-gradient-to-br ${plan.color} rounded-2xl flex items-center justify-center mx-auto mb-4`}>
          <plan.icon className="w-8 h-8 text-white" />
        </div>
        <h3 className="font-poppins font-bold text-2xl text-gray-800 mb-2">{plan.name}</h3>
        <p className="text-gray-600 mb-4">{plan.description}</p>
        
        {currentPrice ? (
          <div className="mb-2">
            <div className="text-5xl font-bold text-vitals-primary mb-2">
              ₹{currentPrice}
              <span className="text-lg text-gray-500 font-normal">/month</span>
            </div>
            {isYearly && originalPrice && (
              <div className="flex items-center justify-center space-x-2">
                <span className="text-sm text-gray-500 line-through">₹{originalPrice}/month</span>
                {savings && (
                  <span className="text-sm text-green-600 font-medium bg-green-100 px-2 py-1 rounded-md">
                    Save {savings}%
                  </span>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="text-5xl font-bold text-purple-600 mb-2">Custom</div>
        )}
        
        <p className="text-sm text-gray-500">
          {isYearly ? "Billed annually" : "Billed monthly"} • {plan.note.split(" • ")[1] || plan.note}
        </p>
      </div>

      <div className="space-y-4 mb-8 flex-1">
        {plan.features.map((feature, index) => (
          <motion.div
            key={feature}
            className="flex items-center space-x-3"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
          >
            <Check className="w-5 h-5 text-vitals-healthy flex-shrink-0" />
            <span className="text-gray-700">{feature}</span>
          </motion.div>
        ))}
      </div>

      <div className="space-y-4">
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button 
            className={`w-full ${getButtonColor()} text-white py-4 rounded-xl font-semibold text-lg transition-colors`}
            data-testid={`cta-${plan.id}`}
          >
            {plan.ctaText}
          </Button>
        </motion.div>
        <p className="text-center text-sm text-gray-500">{plan.note.split(" • ")[0]}</p>
      </div>
    </motion.div>
  );
}
