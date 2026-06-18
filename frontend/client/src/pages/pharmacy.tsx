import { motion } from "framer-motion";
import { Search, ShoppingCart, Truck, ShieldCheck, Percent } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const medicines = [
  {
    id: 1,
    name: "Amlodipine 5mg",
    description: "Blood pressure medication",
    price: 45,
    originalPrice: 60,
    discount: 25,
    image: "https://images.unsplash.com/photo-1584362917165-526a968579e8?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200",
    prescription: false
  },
  {
    id: 2,
    name: "Vitamin D3 1000 IU",
    description: "Bone health supplement",
    price: 120,
    originalPrice: 150,
    discount: 20,
    image: "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200",
    prescription: false
  },
  {
    id: 3,
    name: "Metformin 500mg",
    description: "Diabetes medication",
    price: 35,
    originalPrice: null,
    discount: null,
    image: "https://images.unsplash.com/photo-1631549916768-4119b2e5f926?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200",
    prescription: true
  },
  {
    id: 4,
    name: "Omega-3 Fish Oil",
    description: "Heart health supplement",
    price: 299,
    originalPrice: null,
    discount: null,
    image: "https://images.unsplash.com/photo-1607619056574-7b8d3ee536b2?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200",
    prescription: false,
    badge: "Best Seller"
  },
  {
    id: 5,
    name: "Diclofenac Gel",
    description: "Pain relief gel",
    price: 85,
    originalPrice: null,
    discount: null,
    image: "https://images.unsplash.com/photo-1587854692152-cbe660dbde88?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200",
    prescription: false,
    badge: "Fast Relief"
  },
  {
    id: 6,
    name: "Daily Multivitamin",
    description: "Complete nutrition support",
    price: 199,
    originalPrice: 249,
    discount: 20,
    image: "https://images.unsplash.com/photo-1607619056574-7b8d3ee536b2?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&h=200",
    prescription: false,
    badge: "Popular"
  }
];

export default function Pharmacy() {
  return (
    <div className="pt-20 min-h-screen bg-gradient-to-br from-green-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          className="flex justify-between items-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <h1 className="font-poppins font-bold text-3xl text-gray-800">ePharmacy Shop</h1>
            <p className="text-gray-600 mt-2">Order medicines and health products with ease</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="glass-morphism px-4 py-2 rounded-xl flex items-center">
              <Search className="w-4 h-4 mr-2 text-gray-600" />
              <Input 
                placeholder="Search medicines..." 
                className="bg-transparent border-0 outline-none w-64 focus:ring-0"
                data-testid="search-medicines"
              />
            </div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button className="bg-green-500 text-white hover:bg-green-600 transition-colors relative" data-testid="cart-button">
                <ShoppingCart className="w-4 h-4 mr-2" />
                Cart (3)
                <motion.span
                  className="absolute -top-2 -right-2 w-5 h-5 bg-vitals-critical rounded-full text-xs flex items-center justify-center"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  3
                </motion.span>
              </Button>
            </motion.div>
          </div>
        </motion.div>

        {/* Filters */}
        <motion.div
          className="flex items-center space-x-4 mb-8 flex-wrap gap-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="flex items-center space-x-1 bg-gray-100 rounded-xl p-1">
            <button className="px-4 py-2 rounded-lg bg-white text-vitals-primary shadow-sm font-medium" data-testid="filter-all">
              All Categories
            </button>
            <button className="px-4 py-2 rounded-lg text-gray-600 hover:bg-white hover:shadow-sm transition-all" data-testid="filter-bp">
              Blood Pressure
            </button>
            <button className="px-4 py-2 rounded-lg text-gray-600 hover:bg-white hover:shadow-sm transition-all" data-testid="filter-diabetes">
              Diabetes
            </button>
            <button className="px-4 py-2 rounded-lg text-gray-600 hover:bg-white hover:shadow-sm transition-all" data-testid="filter-vitamins">
              Vitamins
            </button>
          </div>
          <Select>
            <SelectTrigger className="glass-morphism border-0 w-48" data-testid="sort-select">
              <SelectValue placeholder="Sort by Price" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="price-low">Price: Low to High</SelectItem>
              <SelectItem value="price-high">Price: High to Low</SelectItem>
              <SelectItem value="az">A-Z</SelectItem>
              <SelectItem value="popular">Popular</SelectItem>
            </SelectContent>
          </Select>
        </motion.div>

        {/* Product Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mb-12">
          {medicines.map((medicine, index) => (
            <motion.div
              key={medicine.id}
              className="glass-morphism-dark rounded-2xl p-6 hover:shadow-xl transition-all"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ y: -5, scale: 1.02 }}
              data-testid={`medicine-card-${medicine.id}`}
            >
              <img
                src={medicine.image}
                alt={medicine.name}
                className="w-full h-32 object-cover rounded-xl mb-4"
              />
              <div className="space-y-3">
                <div className="flex items-start justify-between">
                  <h3 className="font-semibold text-gray-800">{medicine.name}</h3>
                  {medicine.badge && (
                    <span className="text-sm text-green-600 font-medium bg-green-100 px-2 py-1 rounded-md">
                      {medicine.badge}
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600">{medicine.description}</p>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-vitals-primary">₹{medicine.price}</div>
                    {medicine.originalPrice && (
                      <div className="text-sm text-gray-500 line-through">₹{medicine.originalPrice}</div>
                    )}
                  </div>
                  {medicine.discount && (
                    <div className="text-sm text-green-600 font-medium">{medicine.discount}% off</div>
                  )}
                  {medicine.prescription && (
                    <div className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded text-xs">
                      Prescription Required
                    </div>
                  )}
                </div>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button 
                    className="w-full bg-vitals-primary text-white hover:bg-blue-600 transition-colors font-medium"
                    data-testid={`add-to-cart-${medicine.id}`}
                  >
                    Add to Cart
                  </Button>
                </motion.div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Partner Information */}
        <motion.div
          className="glass-morphism-dark rounded-2xl p-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <div className="text-center mb-6">
            <h3 className="font-poppins font-semibold text-xl text-gray-800 mb-2">Trusted Pharmacy Partners</h3>
            <p className="text-gray-600">We partner with India's leading pharmacies for reliable delivery</p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <motion.div 
              className="text-center"
              whileHover={{ y: -5 }}
            >
              <div className="w-16 h-16 bg-green-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Truck className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold text-gray-800 mb-2">Fast Delivery</h4>
              <p className="text-sm text-gray-600">Same day delivery in major cities, 24-48 hours elsewhere</p>
            </motion.div>
            <motion.div 
              className="text-center"
              whileHover={{ y: -5 }}
            >
              <div className="w-16 h-16 bg-blue-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <ShieldCheck className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold text-gray-800 mb-2">Authentic Medicines</h4>
              <p className="text-sm text-gray-600">100% genuine medications from licensed pharmacies</p>
            </motion.div>
            <motion.div 
              className="text-center"
              whileHover={{ y: -5 }}
            >
              <div className="w-16 h-16 bg-purple-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Percent className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold text-gray-800 mb-2">Best Prices</h4>
              <p className="text-sm text-gray-600">Competitive pricing with exclusive discounts for members</p>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
