# DDoS Detection ML

## Overview

A machine learning application for detecting DDoS (Distributed Denial of Service) attacks. Users can upload CSV/Excel datasets, the system automatically cleans and filters the data, then analyzes it using multiple ML algorithms (Decision Tree, Random Forest, KNN, Naive Bayes, Logistic Regression, LUCID-inspired Neural Network) to compare detection results and identify the best performing model.

### Key Features
- **Multi-format support**: CSV and Excel (.xlsx, .xls) file upload up to 50MB
- **6 ML algorithms**: Including LUCID-inspired Neural Network with convolution filters
- **Attack type classification**: Detects Port Scan, SYN Flood, UDP Flood, ICMP Flood, HTTP Flood, DNS/NTP/LDAP/RDP attacks
- **Vietnamese UI**: Full Vietnamese language interface
- **Detailed explanations**: Formula-based detection with algorithm explanations

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Routing**: Wouter (lightweight React router)
- **State Management**: TanStack React Query for server state and data fetching
- **UI Components**: shadcn/ui component library built on Radix UI primitives
- **Styling**: Tailwind CSS with CSS variables for theming (light/dark mode support)
- **Charts**: Recharts for data visualization (bar charts, radar charts, pie charts)
- **Build Tool**: Vite with React plugin

### Backend Architecture
- **Framework**: Express.js 5 with TypeScript
- **API Pattern**: RESTful JSON API with `/api` prefix
- **Storage**: In-memory storage (MemStorage class) for datasets and analysis results
- **ML Processing**: Custom TypeScript implementations of ML algorithms in `server/ml-algorithms.ts`
- **File Processing**: CSV parsing with automatic data cleaning (handles missing values, duplicates, outliers)

### Data Flow
1. User uploads CSV file → parsed and cleaned on server
2. Dataset stored in memory with preview data
3. User selects ML models to run
4. Server runs analysis and stores results
5. Frontend displays comparison charts and metrics

### Key Design Decisions
- **In-memory storage**: Chosen for simplicity; datasets are session-based and not persisted
- **Server-side ML**: All ML algorithms run on the server to handle larger datasets
- **Monorepo structure**: Client (`client/`), server (`server/`), and shared types (`shared/`) in one repo
- **Path aliases**: `@/` for client source, `@shared/` for shared code

### Project Structure
```
├── client/src/          # React frontend
│   ├── components/      # UI components (shadcn + custom)
│   ├── pages/           # Route pages
│   ├── hooks/           # Custom React hooks
│   └── lib/             # Utilities and query client
├── server/              # Express backend
│   ├── routes.ts        # API endpoints
│   ├── ml-algorithms.ts # ML model implementations
│   └── storage.ts       # In-memory data storage
├── shared/              # Shared TypeScript types
│   └── schema.ts        # Zod schemas for validation
```

## External Dependencies

### Database
- **Drizzle ORM** configured with PostgreSQL dialect (schema in `shared/schema.ts`)
- Database connection via `DATABASE_URL` environment variable
- Currently the app uses in-memory storage, but Drizzle is set up for future database persistence

### Key Libraries
- **Zod**: Schema validation for API requests and data types
- **drizzle-zod**: Generate Zod schemas from Drizzle tables
- **@tanstack/react-query**: Data fetching and caching
- **Radix UI**: Accessible UI primitives (dialog, dropdown, tabs, etc.)
- **Recharts**: React charting library
- **class-variance-authority**: Component variant management

### Build Tools
- **Vite**: Frontend bundling with HMR
- **esbuild**: Server bundling for production
- **tsx**: TypeScript execution for development

### Replit-specific
- `@replit/vite-plugin-runtime-error-modal`: Error overlay in development
- `@replit/vite-plugin-cartographer`: Replit integration