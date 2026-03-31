import { GoogleGenerativeAI } from "@google/generative-ai"
import { generateText } from "ai"

// Mock function to generate test cases when API key is not available
function generateMockTestCases(feature: string) {
  const featureName = feature.length > 30 ? feature.substring(0, 30) + "..." : feature

  return `# Test Cases for: ${featureName}

## Test Case ID: TC-001
**Test Case Title**: Verify Basic Functionality
**Objective**: Ensure the core functionality works as expected
**Preconditions**: User is logged in with valid credentials
**Test Steps**:
1. Navigate to the feature page
2. Enter valid input data
3. Submit the form
4. Verify the results
**Expected Results**: Feature performs the primary function correctly
**Priority**: High
**Test Type**: Functional

## Test Case ID: TC-002
**Test Case Title**: Validate Input Validation
**Objective**: Ensure the system properly validates user inputs
**Preconditions**: User has access to the feature
**Test Steps**:
1. Navigate to the feature page
2. Enter invalid data (e.g., special characters, extremely long text)
3. Submit the form
4. Observe system response
**Expected Results**: System should display appropriate error messages and prevent submission
**Priority**: Medium
**Test Type**: Validation

## Test Case ID: TC-003
**Test Case Title**: Test Error Handling
**Objective**: Verify the system handles errors gracefully
**Preconditions**: System is in a state where errors can occur
**Test Steps**:
1. Create conditions that would trigger an error
2. Execute the feature under these conditions
3. Observe how the system responds
**Expected Results**: System should display user-friendly error messages and recover gracefully
**Priority**: High
**Test Type**: Error Handling

## Test Case ID: TC-004
**Test Case Title**: Performance Under Load
**Objective**: Ensure the feature performs well under heavy usage
**Preconditions**: Test environment capable of simulating load
**Test Steps**:
1. Set up load testing tools
2. Simulate multiple concurrent users
3. Monitor system performance
**Expected Results**: System maintains acceptable response times and doesn't crash
**Priority**: Medium
**Test Type**: Performance

## Test Case ID: TC-005
**Test Case Title**: Mobile Responsiveness
**Objective**: Verify the feature works correctly on mobile devices
**Preconditions**: Access to mobile devices or emulators
**Test Steps**:
1. Access the feature on various mobile devices/screen sizes
2. Test all functionality
3. Check UI layout and usability
**Expected Results**: Feature is fully functional and visually correct on all tested devices
**Priority**: Medium
**Test Type**: UI/Compatibility

Note: These are mock test cases generated for demonstration purposes. In a production environment, the AI would generate more specific test cases tailored to your exact feature requirements.`
}

// Mock function to generate QA support response when API key is not available
function generateMockQASupport(question: string) {
  // Convert question to lowercase for easier matching
  const lowerQuestion = question.toLowerCase();
  
  // Define response categories
  const responses = {
    aes: `# Advanced Encryption Standard (AES)

## Overview
AES is a symmetric encryption algorithm widely used for securing sensitive data. It was established by the U.S. National Institute of Standards and Technology (NIST) in 2001.

## Key Characteristics
- Block cipher algorithm
- Supports key sizes: 128, 192, and 256 bits
- Fixed block size: 128 bits
- Performs multiple rounds of substitution and permutation

## Implementation Modes
- ECB (Electronic Codebook)
- CBC (Cipher Block Chaining)
- CFB (Cipher Feedback)
- OFB (Output Feedback)
- CTR (Counter)

## Best Practices
1. Use strong key generation
2. Implement proper key management
3. Choose appropriate mode of operation
4. Use secure padding schemes
5. Consider authenticated encryption (AES-GCM)

## Common Applications
- File encryption
- Database encryption
- Network security protocols
- Secure communications`,

    security: `# Software Security Fundamentals

## Key Concepts
1. Authentication
2. Authorization
3. Encryption
4. Access Control
5. Input Validation

## Common Security Measures
- SSL/TLS Implementation
- Password Hashing
- JWT Authentication
- CORS Policies
- XSS Prevention
- SQL Injection Protection

## Best Practices
1. Defense in Depth
2. Principle of Least Privilege
3. Regular Security Audits
4. Secure Code Reviews
5. Security Testing`,

    testing: `# Software Testing Guide

## Testing Types
1. Unit Testing
2. Integration Testing
3. System Testing
4. Performance Testing
5. Security Testing

## Best Practices
1. Test Early and Often
2. Maintain Test Independence
3. Follow Arrange-Act-Assert
4. Use Test Automation
5. Implement CI/CD

## Tools and Frameworks
- JUnit, TestNG
- Selenium, Cypress
- JMeter, LoadRunner
- SonarQube
- Test Management Tools`,

    api: `# API Design and Development

## REST Principles
1. Stateless Communication
2. Resource-Based URLs
3. HTTP Methods Usage
4. Status Codes
5. HATEOAS

## Best Practices
1. Version Your APIs
2. Use Proper Authentication
3. Implement Rate Limiting
4. Document Thoroughly
5. Handle Errors Gracefully

## Security Considerations
- API Authentication
- Input Validation
- Rate Limiting
- HTTPS Usage
- Error Handling`,

    default: `# Software Development Concepts

## Key Areas
1. Architecture Patterns
2. Design Principles
3. Testing Methodologies
4. Security Practices
5. Performance Optimization

## Best Practices
1. Write Clean Code
2. Document Properly
3. Test Thoroughly
4. Handle Errors Gracefully
5. Follow Security Guidelines

## Tools and Technologies
- Version Control Systems
- CI/CD Pipelines
- Testing Frameworks
- Monitoring Tools
- Development IDEs`
  };

  // Determine which response to return based on the question
  if (lowerQuestion.includes('aes') || lowerQuestion.includes('encryption')) {
    return responses.aes;
  } else if (lowerQuestion.includes('security') || lowerQuestion.includes('auth')) {
    return responses.security;
  } else if (lowerQuestion.includes('test') || lowerQuestion.includes('qa')) {
    return responses.testing;
  } else if (lowerQuestion.includes('api') || lowerQuestion.includes('rest')) {
    return responses.api;
  } else {
    return responses.default;
  }
}

// Rate limiting configuration
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 1000; // 1 second between requests

// Simple model configuration test
async function testModelConfig(genAI: GoogleGenerativeAI) {
  try {
    // Basic validation that the API key and initialization worked
    return true;
  } catch (error) {
    console.error("Error initializing model:", error);
    return false;
  }
}

export async function POST(request: Request) {
  try {
    const { message, context, model: modelType } = await request.json()

    if (modelType === 'rag') {
      // Call Flask backend RAG endpoint for test case generation
      const flaskRes = await fetch('http://127.0.0.1:5000/api/stories/rag-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message })
      });
      const ragData = await flaskRes.json();
      if (!flaskRes.ok) {
        return Response.json({ error: ragData.error || 'RAG backend error' }, { status: 500 });
      }
      return Response.json({ testCases: ragData.testCases });
    }

    // Check rate limiting
    const now = Date.now();
    if (now - lastRequestTime < MIN_REQUEST_INTERVAL) {
      return Response.json({ 
        error: "Please wait a moment before sending another request." 
      }, { status: 429 });
    }
    lastRequestTime = now;

    // Initialize the Gemini API with configuration
    const apiKey = "AIzaSyDlHeKwQ5qHLr5hV1_1Zkzctl1W5b5oE6Q";
    console.log("Initializing Gemini API with key:", apiKey.substring(0, 8) + "...");
    
    const genAI = new GoogleGenerativeAI(apiKey);

    // Test model configuration
    const isConfigValid = await testModelConfig(genAI);
    if (!isConfigValid) {
      return Response.json({ 
        error: "Failed to verify model configuration. Please check API key and try again." 
      }, { status: 500 });
    }
    
    try {
      console.log("Creating model instance with configuration...");
      const model = genAI.getGenerativeModel({ 
        model: "gemini-2.0-flash",
        generationConfig: {
          temperature: 0.3,
        },
      });

      console.log("Model instance created successfully");
      console.log("Preparing to generate content for message:", message.substring(0, 50) + "...");

      // Prepare the prompt with better context
      const systemPrompt = `As an expert Software Engineer and QA consultant, provide a comprehensive and well-structured response about software development, security, or QA concepts. Format your response using markdown with clear headings, bullet points, and code examples where relevant.

Key areas to cover if applicable:
- Technical concepts and principles
- Best practices and industry standards
- Common challenges and solutions
- Implementation considerations
- Security implications
- Performance aspects`;

      const userMessage = `${systemPrompt}\n\nUser Question: ${message}`;
      
      console.log("Generating content...");
      const result = await model.generateContent(userMessage);
      console.log("Content generated successfully");
      
      const response = await result.response;
      const text = response.text();

      console.log("Response length:", text.length);
      console.log("Response preview:", text.substring(0, 100) + "...");

      if (!text) {
        console.error("No text in Gemini response");
        throw new Error("No response generated from Gemini API");
      }

      return Response.json({ response: text });
    } catch (error: any) {
      console.error("Detailed Gemini API error:", error);
      
      // Handle specific error types
      const errorMessage = error?.message || 'Unknown error occurred';
      
      if (typeof errorMessage === 'string') {
        // API key related errors
        if (errorMessage.toLowerCase().includes('api key')) {
          return Response.json({ 
            error: "Invalid API key. Please check your API key configuration." 
          }, { status: 401 });
        }
        
        // Rate limit errors
        if (errorMessage.toLowerCase().includes('quota') || errorMessage.toLowerCase().includes('rate')) {
          return Response.json({ 
            error: "Please wait a moment and try again." 
          }, { status: 429 });
        }

        // Model not found errors
        if (errorMessage.toLowerCase().includes('not found')) {
          return Response.json({ 
            error: "The specified model is not available. Please check the model configuration." 
          }, { status: 404 });
        }
      }
      
      return Response.json({ 
        error: `Gemini API Error: ${errorMessage}` 
      }, { status: 500 });
    }
  } catch (error: any) {
    console.error("Error in software development support:", error);
    return Response.json({
      error: `Error: ${error?.message || 'An unexpected error occurred'}`,
    }, { status: 500 });
  }
}
