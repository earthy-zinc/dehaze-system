# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dehaze System** is a comprehensive, multi-platform image dehazing system built on deep learning that provides end-to-end solutions for improving image quality affected by haze/fog. The system supports 20+ dehazing algorithms and offers multiple frontend and backend implementations.

## Common Development Commands

### Frontend Development

#### Vue 3 Frontend (dehaze-front-vue)
```bash
cd dehaze-front-vue
# Install dependencies (uses pnpm exclusively)
pnpm install

# Development server (port 5173)
pnpm run dev

# Build for production
pnpm run build

# Run all tests
pnpm test

# Run tests individually
pnpm test:unit                    # Unit tests with Vitest
pnpm test:unit:ui                 # Unit tests with UI
pnpm test:coverage                # Coverage report
pnpm test:e2e                     # E2E tests with Playwright
pnpm test:storybook               # Storybook component tests

# Linting and formatting
pnpm run lint                     # Run all linters
pnpm run lint:eslint              # ESLint
pnpm run lint:prettier            # Prettier
pnpm run lint:stylelint           # Stylelint

# Storybook
pnpm run storybook                # Start Storybook (port 3021)
pnpm run build-storybook          # Build Storybook

# Electron desktop app
pnpm run dev:electron             # Development with Electron
pnpm run build:electron:win       # Build Windows desktop app
pnpm run build:electron:linux     # Build Linux desktop app
```

#### React Frontend (dehaze-front-react)
```bash
cd dehaze-front-react
# Install dependencies
pnpm install

# Development server (port varies)
pnpm run dev

# Build for production
pnpm run build

# Testing (similar structure to Vue)
pnpm test:unit
pnpm test:e2e
pnpm run lint

# Storybook (port 3020)
pnpm run storybook
```

### Backend Development

#### Java Backend (dehaze-java) - Primary Backend
```bash
cd dehaze-java

# Database initialization (required first time)
mysql -u root -p < sql/init.sql

# Build and run
mvn clean install
mvn spring-boot:run               # Runs on port 8989

# Development with auto-reload
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Testing
mvn test
mvn test -Dtest=ClassName        # Run specific test class
```

**Key Configuration** (`src/main/resources/application-dev.yml`):
- Server port: 8989
- MySQL: localhost:3306/dehaze (root/123456)
- Redis: localhost:6379 (password: 123456)
- MongoDB: localhost:27017/dehaze
- MinIO: http://localhost:9000 (admin/12345678)
- API documentation: http://localhost:8989/doc.html

#### Go Backend (dehaze-go) - Alternative Backend
```bash
cd dehaze-go

# Setup and run
go mod download
go run main.go                    # Typically runs on port 8080

# Build
go build -o dehaze-go main.go
./dehaze-go
```

#### Python Algorithm Service (dehaze-python) - Core ML Service
```bash
cd dehaze-python

# Setup environment
conda create -n dehaze_backend python=3.10
conda activate dehaze_backend
pip install -r requirements.txt

# Development
python run.py                     # Runs on port 5000

# Production deployment
gunicorn -w 4 run:app

# Testing (if available)
python -m pytest tests/
```

### Mobile Applications

#### React Native (dehaze-react-native)
```bash
cd dehaze-react-native

# Setup
npm install
npx pod-install ios              # For iOS

# Development
npm run start                     # Start Metro bundler
npx react-native run-android      # Run on Android
npx react-native run-ios          # Run on iOS
```

#### Taro Multi-platform (dehaze-taro)
```bash
cd dehaze-taro

# Development
npm run dev:weapp                 # WeChat Mini Program
npm run dev:h5                    # H5 web version
npm run dev:rn                    # React Native
```

## System Architecture

### Multi-layered Architecture
1. **Client Layer**: Vue 3, React 18, Android Native, React Native, Taro, Electron desktop apps
2. **API Gateway**: Nginx reverse proxy, Spring Cloud Gateway (microservices)
3. **Business Service Layer**: Spring Boot 3.3 (Java), Gin (Go), Flask (Python)
4. **Algorithm Service Layer**: Python service with PyTorch models and 20+ dehazing algorithms
5. **Data Storage Layer**: MySQL 8.0, MongoDB, Redis 6.0+, MinIO/OSS object storage

### Core Algorithm Service (dehaze-python)
- **Main entry point**: `run.py` (Flask development server)
- **Core algorithms**: RIDCP, WPXNet, Dehamer, FFA-Net, AOD-Net, DCP, and 15+ others
- **Framework**: Based on BasicSR with custom implementations
- **GPU Support**: CUDA detection with CPU fallback
- **Inference script**: `dehaze-algorithm/inference_ridcp.py`

### Critical Architecture Patterns
- **Frontend**: Component-based architecture with lazy loading and state management (Pinia/Redux)
- **Java Backend**: Layered architecture (Controller → Service → Mapper → Database)
- **Algorithm Service**: Model factory pattern for dynamic algorithm loading
- **Authentication**: JWT tokens with Redis-based session management
- **File Storage**: Strategy pattern supporting local, MinIO, and Aliyun OSS
- **Real-time Communication**: WebSocket for algorithm progress updates

### Authentication & Security
- **Java Backend**: JWT + RBAC with Spring Security 6
- **Go Backend**: JWT middleware with RBAC
- **Token-based authentication** across all services
- **Redis distributed locks** for concurrency control

### Real-time Features
- **WebSocket**: Progress updates for long-running image processing
- **Vue**: SockJS + StompJS for WebSocket communication
- **Python**: Flask-SocketIO for real-time updates

## Testing Strategy

### Frontend Testing
- **Unit Tests**: Vitest with jsdom environment
- **Component Tests**: Storybook integration tests
- **E2E Tests**: Playwright with multi-browser support (Chrome, Firefox, Safari, Mobile)
- **Coverage**: 80% threshold for lines, functions, branches, statements

### Backend Testing
- **Java**: JUnit 5 + Spring Boot Test
- **Integration Tests**: H2 in-memory database
- **Security Tests**: Spring Security Test

### Test Commands Summary
```bash
# Run all tests (Vue frontend)
cd dehaze-front-vue && pnpm test

# Run E2E tests only
pnpm test:e2e

# Run tests with coverage
pnpm test:coverage

# Run specific Playwright test
npx playwright test image-processing.spec.ts

# Java backend tests
cd dehaze-java && mvn test

# Run single test file (Vue)
pnpm test:unit src/components/ImageProcessor.spec.ts

# Run tests in watch mode (Vue)
pnpm test:unit:ui

# Run Playwright tests with UI
pnpm test:e2e:ui
```

## Development Workflow

### Environment Setup
1. **Install dependencies** for all services you plan to run
2. **Start databases**: MySQL, Redis, MongoDB, MinIO
3. **Initialize database**: Run `sql/init.sql` in the Java backend
4. **Start services**: Backend (Java/Go) � Algorithm service (Python) � Frontend
5. **Verify connectivity**: Check API docs at http://localhost:8989/doc.html

### Port Configuration
- Vue Frontend: 5173
- React Frontend: varies (check vite.config.ts)
- Java Backend: 8989
- Go Backend: 8080 (default)
- Python Algorithm Service: 5000
- Storybook Vue: 3021
- Storybook React: 3020

### Key Development Files
- **Java config**: `dehaze-java/src/main/resources/application-dev.yml`
- **Python Flask app**: `dehaze-python/app/__init__.py`
- **Algorithm implementations**: `dehaze-algorithm/`
- **Vue router**: `dehaze-front-vue/src/router/index.ts`
- **React router**: `dehaze-front-react/src/router/index.tsx`
- **Database init**: `dehaze-java/sql/init.sql` (must run before first startup)
- **Main algorithm inference**: `dehaze-algorithm/inference_ridcp.py`

### Algorithm Development Workflow
```bash
# Test individual algorithm
cd dehaze-algorithm
python inference_ridcp.py -i input.jpg -w model.pth -o output/

# Add new algorithm:
# 1. Create new folder in dehaze-python/algorithm/
# 2. Implement Flask route in dehaze-python/app/api/
# 3. Register algorithm in algorithm factory
```

## Package Management
- **Node.js projects**: pnpm (enforced via `preinstall` script)
- **Java**: Maven
- **Python**: pip + requirements.txt (with conda environment management)
- **Go**: Go modules

## Docker Deployment
All major services include Dockerfile support:
- `dehaze-front-vue/Dockerfile`
- `dehaze-front-react/Dockerfile`
- `dehaze-java/Dockerfile`
- `dehaze-python/Dockerfile` (with GPU support)

## Microservice Variants
- **dehaze-java-cloud**: Basic microservice architecture with Spring Cloud Alibaba
- **dehaze-java-cloud-plus**: Enhanced microservices with AI modules, BPM, CRM, ERP, IoT integration

## Important Notes
- The Java backend is the **primary** and most feature-complete backend implementation
- Python algorithm service requires **GPU acceleration** for optimal performance
- Frontend projects **enforce pnpm** usage via preinstall scripts
- All projects require **Node.js 18+** and **Java 17+**
- Database credentials are **hardcoded for development** - change in production
- The system supports **20+ dehazing algorithms** with dynamic model loading
- **WebSocket** is used for real-time progress updates during image processing
- **Database initialization is mandatory**: Run `mysql -u root -p < dehaze-java/sql/init.sql` before first use
- **MinIO/OSS configuration**: Update file storage settings in application-dev.yml for your environment
- **Algorithm models**: Download required model weights separately (not included in repo)
- 切记！请使用中文和我进行交流。

## Troubleshooting Common Issues
- **Database connection failed**: Check MySQL service and credentials in application-dev.yml
- **Redis connection refused**: Verify Redis service is running on localhost:6379
- **MinIO errors**: Ensure MinIO is running and check bucket permissions
- **GPU not available**: Python algorithm service will fallback to CPU (slower performance)
- **Port conflicts**: Check if ports 5173, 8989, 5000 are already in use