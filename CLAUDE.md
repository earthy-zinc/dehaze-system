# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dehaze System** is a comprehensive, multi-platform image dehazing system built on deep learning that provides end-to-end solutions for improving image quality affected by haze/fog. The system supports 20+ dehazing algorithms and offers multiple frontend and backend implementations.

## Common Development Commands

### Quick Start - Install All JS Dependencies
```bash
PNPM_APPROVE_BUILDS=1 pnpm install -r
```

### Frontend Development

#### Vue 3 Frontend (dehaze-front-vue)
```bash
cd dehaze-front-vue
pnpm install
pnpm run dev                      # Development server (port 5173)
pnpm run build                    # Production build

# Testing
pnpm test                         # Run all tests
pnpm test:unit                    # Unit tests with Vitest
pnpm test:unit:ui                 # Unit tests with UI
pnpm test:coverage                # Coverage report
pnpm test:e2e                     # E2E tests with Playwright
pnpm test:e2e:ui                  # E2E tests with Playwright UI

# Linting
pnpm run lint                     # Run all linters

# Storybook (port 3021)
pnpm run storybook

# Electron desktop app
pnpm run dev:electron
```

#### React Frontend (dehaze-front-react)
```bash
cd dehaze-front-react
pnpm install
pnpm run dev                      # Development server
pnpm test:unit                    # Unit tests
pnpm test:e2e                     # E2E tests
pnpm run storybook                # Storybook (port 3020)
```

#### Flutter App (dehaze_flutter)
```bash
cd dehaze_flutter
flutter pub get
flutter run                       # Run on connected device
flutter build apk                 # Build Android APK
flutter build ios                 # Build iOS
flutter test                      # Run tests
```

#### UniApp (dehaze-uniapp)
```bash
cd dehaze-uniapp
pnpm install
pnpm run dev:h5                   # H5 web version
pnpm run dev:mp-weixin            # WeChat Mini Program
```

#### HarmonyOS (dehaze_harmory)
Uses HarmonyOS DevEco Studio. Check `oh-package.json5` for dependencies.

### Backend Development

#### Java Backend (dehaze-java) - Primary Backend
```bash
cd dehaze-java
mysql -u root -p < sql/init.sql   # Database init (first time only)
mvn clean install
mvn spring-boot:run               # Runs on port 8989
mvn test                          # Run tests
mvn test -Dtest=ClassName         # Run specific test
```

**Dev Config** (`src/main/resources/application-dev.yml`):
- Server: localhost:8989
- MySQL: localhost:3306/dehaze (root/123456)
- Redis: localhost:6379 (password: 123456)
- MongoDB: localhost:27017/dehaze
- MinIO: localhost:9000 (admin/12345678)
- API docs: http://localhost:8989/doc.html

#### Go Backend (dehaze-go)
```bash
cd dehaze-go
go mod download
go run main.go                    # Runs on port 8080
```

#### Python Algorithm Service (dehaze-python) - Core ML
```bash
cd dehaze-python
conda create -n dehaze_backend python=3.10
conda activate dehaze_backend
pip install -r requirements.txt
python run.py                     # Runs on port 5000
gunicorn -w 4 run:app             # Production
```

### Mobile Applications

#### React Native (dehaze-react-native)
```bash
cd dehaze-react-native
npm install && npx pod-install ios
npm run start                     # Metro bundler
npx react-native run-android
npx react-native run-ios
```

#### Taro Multi-platform (dehaze-taro)
```bash
cd dehaze-taro
npm run dev:weapp                 # WeChat Mini Program
npm run dev:h5                    # H5 web version
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
- **Java**: JUnit 5 + Spring Boot Test + H2 in-memory database
- **Python**: pytest (if available)
- **Integration Tests**: API endpoint testing with test containers
- **Security Tests**: Spring Security Test for authentication/authorization

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

### Key Files
- **Java config**: `dehaze-java/src/main/resources/application-dev.yml`
- **Database init**: `dehaze-java/sql/init.sql` (required before first startup)
- **Python Flask app**: `dehaze-python/app/__init__.py`
- **Algorithm implementations**: `dehaze-algorithm/`
- **Main algorithm inference**: `dehaze-algorithm/inference_ridcp.py`
- **Vue router**: `dehaze-front-vue/src/router/index.ts`
- **Vue tests config**: `dehaze-front-vue/vitest.config.ts`

### Algorithm Development
```bash
# Test individual algorithm
cd dehaze-algorithm
python inference_ridcp.py -i input.jpg -w model.pth -o output/

# Add new algorithm:
# 1. Create new folder in dehaze-python/algorithm/
# 2. Implement Flask route in dehaze-python/app/api/
# 3. Register algorithm in algorithm factory
```

### Development Startup Order
1. `mysql -u root -p < dehaze-java/sql/init.sql` (first time only)
2. Start Java backend (port 8989)
3. Start Python algorithm service (port 5000)
4. Start frontend dev server (port 5173)
5. Verify: http://localhost:8989/doc.html

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
- Java backend is the **primary** and most feature-complete implementation
- Python algorithm service benefits from **GPU acceleration** (auto-fallback to CPU)
- Requires **Node.js 18+**, **Java 17+**, **Python 3.8+**
- Frontend projects **enforce pnpm** via preinstall scripts
- **Algorithm models**: Download weights separately (not included in repo)
- **WebSocket** used for real-time progress updates during image processing
- 切记！请使用中文和我进行交流。

## Troubleshooting Common Issues
- **Database connection failed**: Check MySQL service and credentials in application-dev.yml
- **Redis connection refused**: Verify Redis service is running on localhost:6379
- **MinIO errors**: Ensure MinIO is running and check bucket permissions
- **GPU not available**: Python algorithm service will fallback to CPU (slower performance)
- **Port conflicts**: Check if ports 5173, 8989, 5000 are already in use

### Algorithm Service Debugging
- **CUDA detection error**: Check NVIDIA drivers and CUDA installation
- **Model loading failures**: Verify model weights path and file integrity
- **Memory errors**: Reduce batch size or use tiled inference for large images

### Frontend Development Issues
- **pnpm install fails**: Use `PNPM_APPROVE_BUILDS=1 pnpm install -r` for all projects
- **Vite build errors**: Check Node.js version (requires 18+)
- **Playwright tests failing**: Run `npx playwright install` to install browsers

### Backend Development Issues
- **JWT token errors**: Check security.jwt.key in application-dev.yml
- **File upload failures**: Verify MinIO/OSS configuration and permissions
- **Maven build errors**: Check Java version (requires 17+) and Maven settings