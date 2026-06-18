import { motion } from "framer-motion";
import { Activity, Phone } from "lucide-react";
import { Link } from "wouter";

export default function FloatingDashboard() {
  return (
    <motion.div
      className="relative animate-slide-in-right"
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      <motion.div
        className="glass-morphism-dark rounded-3xl p-8 shadow-2xl"
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      >
        {/* Dashboard Header */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="font-poppins font-semibold text-lg text-gray-800">Family Health Overview</h3>
          <div className="flex items-center space-x-2">
            <motion.div
              className="w-3 h-3 bg-vitals-healthy rounded-full"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <span className="text-sm text-gray-600">Live</span>
          </div>
        </div>

        {/* Family Member Cards */}
        <div className="space-y-4">
          {/* Mom */}
          <motion.div
            className="glass-morphism rounded-2xl p-4 hover:shadow-lg transition-all"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gradient-to-br from-pink-400 to-pink-600 rounded-xl flex items-center justify-center">
                  <span className="text-white font-semibold">M</span>
                </div>
                <div>
                  <h4 className="font-medium text-gray-800">Mom</h4>
                  <p className="text-sm text-gray-500">Age 58</p>
                </div>
              </div>
              <div className="text-right">
                <div className="flex items-center space-x-4 text-sm">
                  <div className="text-center">
                    <div className="font-mono text-lg text-vitals-critical">❤️ 78</div>
                    <div className="text-gray-500">BPM</div>
                  </div>
                  <div className="text-center">
                    <div className="font-mono text-lg text-vitals-healthy">🩸 120/80</div>
                    <div className="text-gray-500">BP</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Dad */}
          <motion.div
            className="glass-morphism rounded-2xl p-4 hover:shadow-lg transition-all"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-xl flex items-center justify-center">
                  <span className="text-white font-semibold">D</span>
                </div>
                <div>
                  <h4 className="font-medium text-gray-800">Dad</h4>
                  <p className="text-sm text-gray-500">Age 62</p>
                </div>
              </div>
              <div className="text-right">
                <div className="flex items-center space-x-4 text-sm">
                  <div className="text-center">
                    <div className="font-mono text-lg text-vitals-warning">❤️ 95</div>
                    <div className="text-gray-500">BPM</div>
                  </div>
                  <div className="text-center">
                    <div className="font-mono text-lg text-vitals-warning">🩸 140/90</div>
                    <div className="text-gray-500">BP</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Quick Actions */}
        <div className="mt-6 grid grid-cols-2 gap-3">
          <Link href="/vitals" data-testid="dashboard-view-vitals">
            <motion.button
              className="glass-morphism rounded-xl p-3 text-center hover:bg-blue-50 transition-all w-full"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Activity className="w-6 h-6 mx-auto text-vitals-primary mb-1" />
              <div className="text-sm font-medium text-gray-700">View Vitals</div>
            </motion.button>
          </Link>
          <Link href="/emergency" data-testid="dashboard-emergency">
            <motion.button
              className="glass-morphism rounded-xl p-3 text-center hover:bg-red-50 transition-all w-full"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Phone className="w-6 h-6 mx-auto text-vitals-critical mb-1" />
              <div className="text-sm font-medium text-gray-700">Emergency</div>
            </motion.button>
          </Link>
        </div>
      </motion.div>
    </motion.div>
  );
}
