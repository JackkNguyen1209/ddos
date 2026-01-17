# DDoS Detection ML

## Overview
This project is a machine learning application designed to detect Distributed Denial of Service (DDoS) attacks. It allows users to upload network traffic datasets in CSV or Excel formats. The system processes and cleans the data, then applies various machine learning algorithms, including Decision Tree, Random Forest, KNN, Naive Bayes, Logistic Regression, and a LUCID-inspired Neural Network. The primary goal is to compare the performance of these models in identifying DDoS attacks and classifying over 17 attack types across 7 categories. The application supports both supervised learning (with labeled datasets) and unsupervised anomaly detection (for unlabeled data), offering detailed explanations of detection results and a self-learning capability to improve accuracy over time. The project aims to provide a robust and intuitive tool for cybersecurity analysis.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: React 18 with TypeScript.
- **UI/UX**: Utilizes `shadcn/ui` based on Radix UI, styled with Tailwind CSS, supporting light/dark themes.
- **Data Visualization**: Recharts for interactive data representation.

### Backend
- **Framework**: Express.js 5 with TypeScript.
- **API**: RESTful JSON API.
- **ML Processing**: Custom TypeScript implementations of machine learning algorithms.
- **Data Handling**: In-memory storage for session-based datasets and results, with robust CSV parsing and data cleaning.
- **Security**: Implements Helmet, rate limiting, and CORS.
- **Logging**: Secure audit logging to PostgreSQL.

### Core Features and Design Decisions
- **Monorepo Structure**: Organized into `client/`, `server/`, and `shared/` for modularity.
- **Data Leakage Prevention**: ML pipeline ensures data splitting occurs before preprocessing to maintain model integrity.
- **Reproducibility**: Uses seeded random number generation and precomputed indices for deterministic results across all model evaluations.
- **Advanced ML Practices**:
  - **Train/Validation/Test Split**: Standard 60/20/20 data partitioning.
  - **K-Fold Cross-Validation**: Default 5-fold validation for robust model evaluation.
  - **Hyperparameter Tuning**: Grid Search and Random Search for optimizing model parameters.
- **Expanded Label System**: Supports 130+ built-in labels and allows custom label definitions, enhancing attack classification.
- **User Feedback & Review**: System for users to correct analysis results, contributing to the self-learning mechanism.
- **Schema Detection & Model Routing**: Automatically identifies dataset schemas (e.g., CICFlowMeter vs. Event/Log), normalizes columns, and recommends appropriate ML models.
- **Ensemble Learning**: Implements a Voting Classifier to combine predictions from multiple models.
- **Performance**: Multipart file uploads with `multer`, PapaParse for CSV, and persistent PostgreSQL storage for metadata and audit logs.

## External Dependencies

- **Database**: PostgreSQL with Drizzle ORM for persistent storage of metadata, analysis results, and audit logs.
- **Validation**: Zod for schema validation.
- **Data Fetching**: `@tanstack/react-query` for server state management.
- **UI Components**: Radix UI primitives, `shadcn/ui`.
- **Charting**: Recharts.
- **Build Tools**: Vite (frontend), esbuild (server), tsx (development).
- **Security Middleware**: Helmet.
- **CSV Parsing**: PapaParse.
- **File Upload**: Multer.
- **Testing**: Vitest for unit testing.

## Testing

Run the ML pipeline determinism tests with:
```bash
npx vitest run --root=. server/ml-pipeline.test.ts
```

The test suite verifies:
- Global seed management (setGlobalSeed/getGlobalSeed)
- Train/validation/test split reproducibility (makeSplitIndices)
- K-fold cross-validation fold generation (makeKFolds)
- Cross-validation result determinism (kFoldCrossValidation)
- Anomaly detection reproducibility (runAnomalyDetection)