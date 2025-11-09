# Developer Guide

## Development Environment Setup

### Required Tools
- Git (version 2.30+)
- Docker Desktop
- VS Code or IntelliJ IDEA
- Node.js 18+ or Python 3.9+
- PostgreSQL 14+

### Getting Started
1. Clone the repository: `git clone https://github.com/company/main-repo.git`
2. Install dependencies: `npm install` or `pip install -r requirements.txt`
3. Copy environment file: `cp .env.example .env`
4. Start development server: `npm run dev` or `python manage.py runserver`

## Code Standards

### Git Workflow
- Create feature branches from `main`: `git checkout -b feature/my-feature`
- Commit messages follow conventional commits format
- Example: `feat: add user authentication`, `fix: resolve login bug`
- Keep commits small and focused
- Rebase before merging to keep history clean

### Code Review Process
1. Create pull request when feature is ready
2. Request review from at least 2 team members
3. All CI checks must pass (tests, linting, security scans)
4. Address all review comments
5. Squash commits before merging
6. Delete feature branch after merge

### Testing Requirements
- Unit test coverage must be >80%
- Write tests before code (TDD encouraged)
- Integration tests for API endpoints
- E2E tests for critical user flows
- Run tests locally before pushing: `npm test` or `pytest`

## Deployment Process

### Environments
- **Development**: auto-deploy from `develop` branch
- **Staging**: manual promotion from development
- **Production**: requires approval from tech lead

### Deployment Steps
1. Merge feature branch to `develop`
2. CI/CD pipeline runs automatically
3. If tests pass, deploys to development environment
4. QA team tests in development
5. Promote to staging for final testing
6. Create release PR to `main` branch
7. After approval, deploy to production

### Monitoring
- Use Grafana for metrics: https://grafana.company.com
- Check logs in DataDog: https://app.datadoghq.com
- Set up alerts for errors and performance issues
- On-call rotation uses PagerDuty

## API Development

### RESTful API Guidelines
- Use standard HTTP methods: GET, POST, PUT, DELETE, PATCH
- Return appropriate status codes: 200, 201, 400, 404, 500
- Use JSON for request and response bodies
- Version APIs: `/api/v1/`, `/api/v2/`
- Document all endpoints in Swagger/OpenAPI

### Authentication
- Use JWT tokens for authentication
- Include token in Authorization header: `Bearer <token>`
- Tokens expire after 24 hours
- Refresh tokens valid for 30 days
- Implement rate limiting (100 requests per minute)

### Error Handling
```python
# Standard error response format
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  }
}
```

## Database

### Migrations
- Create migrations for all schema changes
- Never edit migration files directly
- Test migrations on development first
- Backup production before running migrations
- Migrations must be reversible

### Best Practices
- Use database indexes for frequently queried columns
- Avoid N+1 queries (use joins or eager loading)
- Keep transactions small and fast
- Use connection pooling
- Regular database backups (automated daily)

## Security

### Secure Coding Practices
- Never commit secrets or API keys
- Use environment variables for configuration
- Validate and sanitize all user input
- Use parameterized queries to prevent SQL injection
- Implement CSRF protection
- Set proper CORS headers
- Keep dependencies up to date

### Secrets Management
- Use AWS Secrets Manager or HashiCorp Vault
- Rotate secrets regularly (every 90 days)
- Different secrets for each environment
- Access secrets through SDK, never hardcode

## Performance

### Optimization Guidelines
- Cache frequently accessed data (Redis)
- Use CDN for static assets
- Compress responses (gzip)
- Lazy load images and components
- Database query optimization
- Monitor response times (target: <200ms)

### Load Testing
- Use k6 or JMeter for load testing
- Test before major releases
- Target: 1000 concurrent users
- Monitor CPU, memory, and database performance

## Documentation

### Code Documentation
- Write clear comments for complex logic
- Document all public APIs
- Keep README files up to date
- Use JSDoc or Python docstrings
- Include examples in documentation

### Wiki Pages
- Architecture diagrams on Confluence
- Runbooks for common issues
- Onboarding guides for new developers
- Decision records (ADRs) for major changes

## Support

### Getting Help
- Slack channel: #engineering-help
- Weekly office hours: Fridays 2-4 PM
- Mentor program for new developers
- Documentation: docs.company.com

### Reporting Issues
- Create Jira ticket for bugs
- Include reproduction steps
- Attach relevant logs and screenshots
- Assign to appropriate team

## Code Snippets

### Python API Example
```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: int):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### JavaScript Example
```javascript
// Fetch data with error handling
async function fetchUserData(userId) {
  try {
    const response = await fetch(`/api/v1/users/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch user');
    }
    return await response.json();
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

## Contact

- Tech Lead: tech.lead@company.com
- DevOps Team: devops@company.com
- Security Team: security@company.com
