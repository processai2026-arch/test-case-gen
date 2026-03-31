"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { api, TestCase, TestCaseResponse } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, Download, ArrowLeft, Info, CheckCircle2, AlertCircle, Brain, Triangle, Eye, FileText } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Chatbot } from "@/components/chatbot";

export default function StoryTestCasesPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const projectId = searchParams.get('project_id');
  const storyId = typeof params?.storyId === 'string' ? params.storyId : Array.isArray(params?.storyId) ? params.storyId[0] : '';
  const [testCases, setTestCases] = useState<TestCase[]>([]);
  const [storyDescription, setStoryDescription] = useState<string | null>(null);
  const [storyContent, setStoryContent] = useState<string | null>(null);
  const [storyProject, setStoryProject] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<string>("");
  const [selectedTestCase, setSelectedTestCase] = useState<TestCase | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [showStoryContent, setShowStoryContent] = useState(false);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([
    {
      role: "assistant",
      content:
        "Hello! I'm your AI Test Case Assistant. I can help you analyze and improve the test cases for this story. What would you like to know?",
    },
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'gemini' | 'rag'>('rag');
  const [impactedTestCases, setImpactedTestCases] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await api.getTestCases(storyId);
        setTestCases(response.testcases);
        setStoryDescription(response.storyDescription);
        // Fetch story content
        const storyResponse = await fetch(`http://127.0.0.1:5000/api/stories/${storyId}`);
        if (storyResponse.ok) {
          const storyData = await storyResponse.json();
          console.log('Story data response:', storyData); // Debug log
          setStoryContent(storyData.document_content);
          setStoryProject(storyData.project_id);
        } else {
          console.error('Failed to fetch story:', storyResponse.status);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch test cases');
      } finally {
        setLoading(false);
      }
    };

    if (storyId) {
      fetchData();
    }
  }, [storyId]);

  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString());
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const loadImpactedTestCases = async () => {
      if (!projectId) {
        console.error('Project ID is missing');
        return;
      }

      try {
        const data = await api.fetchImpactedTestCases(storyId, projectId);
        if (data) {
          setImpactedTestCases(data);
        }
      } catch (error) {
        console.error('Error loading impacted test cases:', error);
      }
    };

    if (projectId) {
      loadImpactedTestCases();
    }
  }, [storyId, projectId]);

  const handleViewDetails = (testCase: TestCase) => {
    setSelectedTestCase(testCase);
    setIsDialogOpen(true);
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/stories/testcases/download/${storyId}`);
      if (!response.ok) throw new Error('Download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `test_cases_${storyId}.xlsx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download error:', error);
    }
  };

  const toggleCardExpand = (e: React.MouseEvent, cardId: string) => {
    e.stopPropagation();
    setExpandedCards(prev => {
      const newSet = new Set(prev);
      if (newSet.has(cardId)) {
        newSet.delete(cardId);
      } else {
        newSet.add(cardId);
      }
      return newSet;
    });
  };

  const handleSendMessage = async (message: string) => {
    if (!message.trim() || isGenerating) return;

    // Add user message to chat
    setChatMessages((prev) => [...prev, { role: "user", content: message }]);
    setIsGenerating(true);

    try {
      const response = await fetch("/api/generate-test-cases", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          context: selectedModel === 'rag' ? "Generate comprehensive test cases for the following feature or user story" : "Provide QA support and guidance",
          model: selectedModel,
          storyId: params.storyId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to generate response");
      }

      if (data.error) {
        throw new Error(data.error);
      }

      // If RAG, display structured test cases
      if (selectedModel === 'rag' && data.response) {
        const { parsed, cleaned } = cleanAndParseLLMResponse(data.response);
        if (parsed && Array.isArray(parsed)) {
          setChatMessages((prev) => [
            ...prev,
            { role: "assistant", content: formatRagTestCases(parsed) }
          ]);
        } else {
          setChatMessages((prev) => [
            ...prev,
            { role: "assistant", content: cleaned || '[No output from LLM]' }
          ]);
        }
      } else if (selectedModel === 'rag' && data.testCases) {
        setChatMessages((prev) => [...prev, { role: "assistant", content: formatRagTestCases(data.testCases) }]);
      } else {
        // For Gemini mode, display the response directly
        setChatMessages((prev) => [...prev, { role: "assistant", content: data.response }]);
      }
    } catch (error) {
      console.error("Error generating response:", error);
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: selectedModel === 'rag' 
            ? "I'm using mock test cases for demonstration. In production, you would connect to the RAG backend."
            : "I'm using mock QA support for demonstration. In production, you would connect to the Gemini API.",
        },
      ]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="fixed top-0 left-0 w-full z-50 bg-background shadow-sm border-b border-border px-4 md:px-8 py-4">
        <div className="flex justify-between items-center">
          <div className="flex-shrink-0">
            <img src="/Logo-New.svg" alt="Innova Solutions" className="h-12 w-auto" />
          </div>
          <div className="text-center">
            <h1 className="text-3xl font-bold text-primary">Test Forge AI</h1>
            <p className="text-sm text-muted-foreground font-semibold mt-1">Powered by Gen AI</p>
          </div>
          <div className="flex-shrink-0 flex items-center space-x-4">
            <div className="text-right">
              <div className="text-lg font-bold text-foreground">
                {currentTime || 'Loading...'}
              </div>
              <div className="text-sm text-muted-foreground">{new Date().toLocaleDateString()}</div>
            </div>
            <div className="bg-blue-500 text-white px-4 py-2 rounded-full flex items-center space-x-2">
              <div className="w-6 h-6 bg-white/20 rounded-full flex items-center justify-center">
                <Triangle className="h-3 w-3 fill-white" />
              </div>
              <div className="text-sm font-bold">
                <div className="text-xs opacity-90">Team</div>
                <div>DELTA</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-[95%] mx-auto py-8 pt-32">
        <Card className="shadow-lg border border-border rounded-xl overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-muted/50 to-background border-b border-border">
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Button
                    variant="outline"
                    onClick={() => router.back()}
                    className="flex items-center gap-2 border-border hover:bg-muted transition-colors"
                  >
                    <ArrowLeft className="h-4 w-4" />
                    Back to Stories
                  </Button>
                  <div className="h-6 w-px bg-border"></div>
                  <div className="flex items-center gap-2">
                    <div className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-full text-sm font-medium flex items-center gap-1">
                      <CheckCircle2 className="h-4 w-4" />
                      {testCases.length} Test Cases
                    </div>
                    <div className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300 rounded-full text-sm font-medium flex items-center gap-1">
                      <Brain className="h-4 w-4" />
                      AI Generated
                    </div>
                    {storyProject && (
                      <div className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 rounded-full text-sm font-medium flex items-center gap-1">
                        <FileText className="h-4 w-4" />
                        Project: {storyProject}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setShowStoryContent(true)}
                    className="flex items-center gap-2 border-border hover:bg-muted transition-colors"
                  >
                    <FileText className="h-4 w-4" />
                    View Story Content
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleDownload}
                    className="flex items-center gap-2 border-border hover:bg-muted transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    Download Test Cases
                  </Button>
                </div>
              </div>
              
              <div className="mt-2">
                <CardTitle className="text-xl font-semibold text-primary mb-3">
                  Test Cases for Story {storyId}
                </CardTitle>
                {storyDescription && (
                  <p className="text-foreground text-sm leading-relaxed">
                    {storyDescription}
                  </p>
                )}
              </div>
            </div>
          </CardHeader>

          <CardContent className="p-0 bg-card">
            {loading ? (
              <div className="p-8 text-center text-blue-600 font-semibold">Loading...</div>
            ) : error ? (
              <div className="p-8 text-center text-red-600 font-semibold">{error}</div>
            ) : testCases.length === 0 ? (
              <div className="p-8 text-center text-muted-foreground">No test cases found for this story.</div>
            ) : (
              <div className="grid grid-cols-2 gap-6 p-6">
                {testCases.map((tc, idx) => (
                  <Card 
                    key={tc.test_case_id || tc.id || idx} 
                    className="border border-border rounded-lg overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => handleViewDetails(tc)}
                  >
                    <div className="p-4 bg-muted border-b border-border">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center text-blue-800 dark:text-blue-300 font-semibold">
                            {idx + 1}
                          </div>
                          <div>
                            <div className="font-semibold text-primary">{tc.title}</div>
                            <div className="text-muted-foreground text-sm">
                              ID: <span className="font-mono">{tc.test_case_id || tc.id || "N/A"}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            tc.priority === 'high' ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300' :
                            tc.priority === 'medium' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                            'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                          }`}>
                            {tc.priority || 'low'} priority
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleViewDetails(tc);
                            }}
                            className="text-blue-600 hover:text-blue-800 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Story Content Dialog */}
      <Dialog open={showStoryContent} onOpenChange={setShowStoryContent}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold text-primary">
              Story Content
            </DialogTitle>
            <DialogDescription className="text-sm text-muted-foreground">
              ID: {storyId}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Story Description</h3>
              <div className="bg-muted rounded-lg p-4">
                <p className="text-foreground">{storyDescription}</p>
              </div>
            </div>
            
            {storyContent && (
              <div>
                <h3 className="text-sm font-semibold text-foreground mb-2">Story Content</h3>
                <div className="bg-muted rounded-lg p-4">
                  <p className="text-foreground whitespace-pre-wrap">{storyContent}</p>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Test Case Details Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold text-primary">
              {selectedTestCase?.title}
            </DialogTitle>
            <DialogDescription className="text-sm text-muted-foreground">
              ID: {selectedTestCase?.test_case_id || selectedTestCase?.id || "N/A"}
            </DialogDescription>
          </DialogHeader>
          
          {selectedTestCase && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-semibold text-foreground mb-2">Test Steps</h3>
                    <div className="space-y-2">
                      {selectedTestCase.steps?.map((step, stepIdx) => (
                        <div key={stepIdx} className="flex gap-3 p-3 bg-muted rounded-lg">
                          <div className="w-6 h-6 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center text-blue-800 dark:text-blue-300 font-semibold flex-shrink-0">
                            {stepIdx + 1}
                          </div>
                          <div className="text-sm text-foreground">{step}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-semibold text-foreground mb-2">Expected Results</h3>
                    <div className="space-y-2">
                      {selectedTestCase?.expected_results ? (
                        // Multiple expected results
                        selectedTestCase.expected_results.map((result, resultIdx) => (
                          <div key={resultIdx} className="flex gap-3 p-3 bg-muted rounded-lg">
                            <div className="w-6 h-6 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center text-green-800 dark:text-green-300 font-semibold flex-shrink-0">
                              {resultIdx + 1}
                            </div>
                            <div className="text-sm text-foreground">{result}</div>
                          </div>
                        ))
                      ) : selectedTestCase?.expected_result ? (
                        // Single expected result
                        <div className="flex gap-3 p-3 bg-muted rounded-lg">
                          <div className="w-6 h-6 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center text-green-800 dark:text-green-300 font-semibold flex-shrink-0">
                            1
                          </div>
                          <div className="text-sm text-foreground">{selectedTestCase.expected_result}</div>
                        </div>
                      ) : (
                        <div className="text-sm text-muted-foreground italic">No expected results defined</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-border">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-foreground">Priority:</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      selectedTestCase.priority === 'high' ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300' :
                      selectedTestCase.priority === 'medium' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                      'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                    }`}>
                      {selectedTestCase.priority || 'low'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-foreground">Generated by:</span>
                    <span className="px-2 py-1 rounded text-xs font-medium bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300 flex items-center gap-1">
                      <Brain className="h-3 w-3" />
                      AI
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {impactedTestCases && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Impact Analysis by AI</h3>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <Card>
              <CardHeader>
                <CardTitle>Total Test Cases</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{impactedTestCases.total_test_cases}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Impacted Test Cases</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{impactedTestCases.impacted_test_cases_count}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Impact Percentage</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{impactedTestCases.impact_percentage}%</p>
              </CardContent>
            </Card>
          </div>

          <div className="mb-4">
            <h4 className="text-md font-semibold mb-2">Severity Summary</h4>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-red-100 p-4 rounded-lg">
                <p className="text-red-700">High: {impactedTestCases.severity_summary.high}</p>
              </div>
              <div className="bg-yellow-100 p-4 rounded-lg">
                <p className="text-yellow-700">Medium: {impactedTestCases.severity_summary.medium}</p>
              </div>
              <div className="bg-green-100 p-4 rounded-lg">
                <p className="text-green-700">Low: {impactedTestCases.severity_summary.low}</p>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-md font-semibold mb-2">Impacted Test Cases Details</h4>
            <div className="space-y-4">
              {impactedTestCases.impacted_test_cases.map((testCase: any) => (
                <Card key={testCase.test_case_id}>
                  <CardHeader>
                    <CardTitle>{testCase.original_title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <p><strong>Test Case ID:</strong> {testCase.test_case_id}</p>
                      <p><strong>Expected Result:</strong> {testCase.original_expected_result}</p>
                      <div>
                        <strong>Steps:</strong>
                        <ul className="list-disc pl-6">
                          {testCase.original_steps.map((step: string, index: number) => (
                            <li key={index}>{step}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <strong>Impacts:</strong>
                        <div className="space-y-2 mt-2">
                          {testCase.impacts.map((impact: any, index: number) => (
                            <div key={index} className="border p-2 rounded">
                              <p><strong>Impacting Story:</strong> {impact.impacting_story_description}</p>
                              <p><strong>Severity:</strong> {impact.impact_severity}</p>
                              <p><strong>Priority:</strong> {impact.impact_priority}</p>
                              <p><strong>Impact Details:</strong> {impact.impact_details}</p>
                              <p><strong>Created On:</strong> {new Date(impact.impact_created_on).toLocaleString()}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      )}

      <Chatbot
        onSendMessage={handleSendMessage}
        chatMessages={chatMessages}
        isGenerating={isGenerating}
        selectedModel={selectedModel}
        onModelChange={(model) => setSelectedModel(model)}
      />
    </div>
  );
}

function cleanAndParseLLMResponse(response: string) {
  // Remove triple backticks and ```json if present
  let cleaned = response.trim()
  if (cleaned.startsWith('```json')) cleaned = cleaned.slice(7)
  if (cleaned.startsWith('```')) cleaned = cleaned.slice(3)
  if (cleaned.endsWith('```')) cleaned = cleaned.slice(0, -3)
  cleaned = cleaned.trim()
  try {
    const parsed = JSON.parse(cleaned)
    return { parsed, cleaned }
  } catch {
    return { parsed: null, cleaned }
  }
}

function formatRagTestCases(testCases: any[]) {
  return testCases.map((tc, idx) => {
    const steps = Array.isArray(tc.steps) ? tc.steps.join('\n') : tc.steps
    return `Test Case ${idx + 1}: ${tc.title}\nSteps:\n${steps}\nExpected Result: ${tc.expected_result}\n`
  }).join('\n\n')
} 