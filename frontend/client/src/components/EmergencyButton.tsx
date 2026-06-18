import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

export default function EmergencyButton() {
  const [tapCount, setTapCount] = useState(0);
  const [buttonText, setButtonText] = useState("🚑 Call Emergency Services");
  const [isActivated, setIsActivated] = useState(false);

  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (tapCount === 1) {
      setButtonText("🚑 Tap 2 more times to confirm");
      timer = setTimeout(() => {
        setTapCount(0);
        setButtonText("🚑 Call Emergency Services");
      }, 3000);
    } else if (tapCount === 2) {
      setButtonText("🚨 Tap once more to activate!");
      timer = setTimeout(() => {
        setTapCount(0);
        setButtonText("🚑 Call Emergency Services");
      }, 3000);
    } else if (tapCount >= 3) {
      setButtonText("🚨 Emergency Services Contacted!");
      setIsActivated(true);
      timer = setTimeout(() => {
        setTapCount(0);
        setButtonText("🚑 Call Emergency Services");
        setIsActivated(false);
      }, 5000);
    }

    return () => clearTimeout(timer);
  }, [tapCount]);

  const handleEmergencyClick = () => {
    if (!isActivated) {
      setTapCount(prev => prev + 1);
    }
  };

  const getButtonColor = () => {
    if (isActivated) return "bg-green-600 hover:bg-green-700";
    if (tapCount === 2) return "bg-red-700 hover:bg-red-800";
    return "bg-red-600 hover:bg-red-700";
  };

  return (
    <motion.div
      whileHover={{ scale: isActivated ? 1 : 1.05 }}
      whileTap={{ scale: isActivated ? 1 : 0.95 }}
    >
      <Button
        onClick={handleEmergencyClick}
        disabled={isActivated}
        className={`w-full ${getButtonColor()} text-white py-4 rounded-xl font-bold text-lg transition-all emergency-glow ${
          tapCount > 0 && !isActivated ? "animate-pulse" : ""
        }`}
        data-testid="critical-emergency-button"
      >
        {buttonText}
      </Button>
    </motion.div>
  );
}
