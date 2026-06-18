# VitalsBridge Healthcare Platform

## Overview

VitalsBridge is a comprehensive healthcare SaaS platform designed for remote family health monitoring in Indian households. The platform combines real-time vitals tracking, smart emergency alerts, AI health assistance, and integrated pharmacy services to provide "Peace of Health for Every Heartbeat You Care About."

The application serves two primary user types: medical professionals through a clinical-grade doctor dashboard and families through an intuitive caregiver interface. It focuses on making preventive healthcare visible, accessible, and human-centered for Indian families by supporting multiple languages and integrating with popular health devices.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript using Vite as the build tool
- **Routing**: Wouter for client-side routing with pages for home, dashboards, emergency, pharmacy, pricing, AI assistant, vitals, and alerts
- **UI Library**: Shadcn/ui components built on Radix UI primitives with Tailwind CSS for styling
- **State Management**: TanStack React Query for server state management and caching
- **Styling**: Tailwind CSS with custom CSS variables for VitalsBridge branding (primary blue #007BFF, health status colors)
- **Typography**: Poppins for headings, Inter for body text, Roboto Mono for vitals display
- **Animations**: Framer Motion for micro-interactions and page transitions

### Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **API Design**: RESTful API structure with `/api` prefix for all endpoints
- **Development Server**: Custom Vite integration for hot module replacement in development
- **Storage Interface**: Abstract storage layer with in-memory implementation (MemStorage) for development

### Data Storage Solutions
- **Database**: PostgreSQL configured through Drizzle ORM
- **ORM**: Drizzle with TypeScript-first schema definitions
- **Migrations**: Drizzle Kit for database schema management
- **Connection**: Neon Database serverless PostgreSQL via environment variable
- **Schema**: Shared schema definitions between client and server with Zod validation

### Authentication and Authorization
- **User Model**: Basic user schema with username/password fields and UUID primary keys
- **Session Management**: Ready for implementation with connect-pg-simple for PostgreSQL session storage
- **Validation**: Zod schemas for runtime type checking and API request validation

### External Dependencies
- **Health Device Integration**: Planned integration with iHealth, Dozee, Omron, and Wellue devices for vitals monitoring
- **Pharmacy Services**: Integration capabilities for 1mg and PharmEasy for medication delivery
- **Communication**: WhatsApp and SMS APIs for emergency alerts and family notifications
- **Maps/Location**: GPS integration for emergency location sharing
- **AI Services**: Multilingual support for Telugu, Hindi, and English with voice/text analysis
- **Charts/Visualization**: Recharts library for health trend visualization and vitals tracking

### Design System
- **Glass Morphism**: Custom CSS classes for modern glass-like UI effects
- **Color Scheme**: Healthcare-focused palette with status indicators (healthy green, warning yellow, critical red)
- **Responsive Design**: Mobile-first approach with breakpoint considerations
- **Accessibility**: Radix UI primitives ensure keyboard navigation and screen reader support
- **Component Library**: Comprehensive set of reusable UI components following consistent design patterns

The architecture emphasizes modularity, type safety, and scalability while maintaining a focus on healthcare-specific requirements like real-time monitoring, emergency response, and multilingual support for Indian families.