"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronUp, Download, ArrowLeft, Info, CheckCircle2, AlertCircle, Brain, Triangle, Eye, FileText } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Chatbot } from "@/components/chatbot";

interface ImpactedTestCase {
  id: string;
  modified_title: string;
  modified_test_case_id: string;
  severity: 'high' | 'medium' | 'low';
  priority: 'high' | 'medium' | 'low';
  original_priority: string;
  impacts: any[];
  original_story_id: string;
  original_title: string;
  original_steps?: string[];
  original_expected_result?: string;
}

export default function StoryImpactAnalysisPage() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const projectId = searchParams.get('project_id');
  const storyId = typeof params?.storyId === 'string' ? params.storyId : Array.isArray(params?.storyId) ? params.storyId[0] : '';
  const [impactedTestCases, setImpactedTestCases] = useState<ImpactedTestCase[]>([]);
  const [storyDescription, setStoryDescription] = useState<string | null>(null);
  const [storyContent, setStoryContent] = useState<string | null>(null);
  const [storyProject, setStoryProject] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingOriginal, setLoadingOriginal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<string>("");
  const [selectedTestCase, setSelectedTestCase] = useState<ImpactedTestCase | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [showStoryContent, setShowStoryContent] = useState(false);
  const [showOriginalTestCase, setShowOriginalTestCase] = useState(false);
  const [showFullContent, setShowFullContent] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([
    {
      role: "assistant",
      content:
        "Hello! I'm your AI Test Case Assistant. I can help you analyze this story and generate test cases. What would you like to know?",
    },
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'gemini' | 'rag'>('rag');
  const [summaryData, setSummaryData] = useState<{
    total_test_cases: number;
    impacted_test_cases_count: number;
    impact_percentage: number;
    severity_summary: {
      high: number;
      medium: number;
      low: number;
    }
  }>({
    total_test_cases: 0,
    impacted_test_cases_count: 0,
    impact_percentage: 0,
    severity_summary: {
      high: 0,
      medium: 0,
      low: 0
    }
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        if (!projectId) {
          throw new Error('Project ID is required');
        }

        // Fetch impacted test cases using the correct endpoint
        const response = await fetch(`http://127.0.0.1:5000/api/stories/impacts/story-test-cases/${storyId}?project_id=${projectId}`);
        if (!response.ok) throw new Error('Failed to fetch impacted test cases');
        const data = await response.json();
        
        // Transform the data to match our interface and filter only impacted test cases
        const transformedTestCases = data.impacted_test_cases
          .filter((tc: any) => tc.impacts && tc.impacts.length > 0) // Only keep test cases with impacts
          .map((tc: any) => ({
            id: tc.test_case_id,
            original_title: tc.original_title || '',
            modified_title: tc.impacts[0]?.impact_analysis_json ? JSON.parse(tc.impacts[0].impact_analysis_json)?.modified_test_case?.title || `Modified ${tc.original_title}` : `Modified ${tc.original_title}`,
            modified_test_case_id: `${tc.test_case_id}-MOD-${tc.impacts.length}`,
            severity: tc.impacts[0]?.impact_severity || 'low',
            priority: tc.impacts[0]?.impact_priority || 'low',
            original_priority: tc.impacts[0]?.impact_analysis_json ? JSON.parse(tc.impacts[0].impact_analysis_json)?.modified_test_case?.priority || 'Medium' : 'Medium',
            impacts: tc.impacts,
            original_story_id: storyId,
            original_steps: [],
            original_expected_result: ''
          }));
        
        setImpactedTestCases(transformedTestCases);
        
        // Set summary data
        setSummaryData({
          total_test_cases: parseInt(data.total_test_cases),
          impacted_test_cases_count: parseInt(data.impacted_test_cases_count),
          impact_percentage: parseFloat(data.impact_percentage),
          severity_summary: {
            high: parseInt(data.severity_summary.high),
            medium: parseInt(data.severity_summary.medium),
            low: parseInt(data.severity_summary.low)
          }
        });
        
        // Fetch story details
        const storyResponse = await fetch(`http://127.0.0.1:5000/api/stories/${storyId}`);
        if (storyResponse.ok) {
          const storyData = await storyResponse.json();
          setStoryDescription(storyData.description);
          setStoryContent(storyData.document_content);
          setStoryProject(storyData.project_id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch impact analysis');
      } finally {
        setLoading(false);
      }
    };

    if (storyId && projectId) {
      fetchData();
    }
  }, [storyId, projectId]);

  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString());
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleViewDetails = (testCase: ImpactedTestCase) => {
    setSelectedTestCase(testCase);
    setIsDialogOpen(true);
  };

  // Function to fetch original test case details
  const fetchOriginalTestCase = async (testCase: ImpactedTestCase) => {
    try {
      setLoadingOriginal(true);
      console.log('Fetching original test case:', {
        original_test_case_id: testCase.id,
        original_story_id: storyId
      });
      
      // Updated endpoint to include story ID
      const response = await fetch(`http://127.0.0.1:5000/api/stories/${storyId}/test-cases/${testCase.id}`);
      if (!response.ok) throw new Error('Failed to fetch original test case');
      const data = await response.json();
      
      console.log('Original test case data:', data);

      // Update the selected test case with original data
      setSelectedTestCase(prev => prev ? {
        ...prev,
        original_steps: data.test_steps || data.steps || [],
        original_expected_result: data.expected_result || '',
        original_title: data.title || data.test_title || prev.original_title || ''
      } : null);
    } catch (err) {
      console.error('Error fetching original test case:', err);
    } finally {
      setLoadingOriginal(false);
    }
  };

  // Add effect to fetch original test case when showing it
  useEffect(() => {
    if (showOriginalTestCase && selectedTestCase?.id) {
      fetchOriginalTestCase(selectedTestCase);
    }
  }, [showOriginalTestCase]);

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
      {/* Header */}
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

      {/* Main Content */}
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
                    <div className="px-3 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-300 rounded-full text-sm font-medium flex items-center gap-1">
                      <Brain className="h-4 w-4" />
                      {impactedTestCases.length} Modified Test Cases
                    </div>
                    {storyProject && (
                      <div className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 rounded-full text-sm font-medium flex items-center gap-1">
                        <FileText className="h-4 w-4" />
                        Project: {storyProject}
                      </div>
                    )}
                  </div>
                </div>
              </div>
                {storyDescription && (
                <div className="text-muted-foreground">
                  <span className="font-medium text-foreground">Story Description: </span>
                    {storyDescription}
                </div>
                )}
            </div>
          </CardHeader>

          <CardContent>
            {loading ? (
              <div className="p-8 text-center text-blue-600 font-semibold">Loading...</div>
            ) : error ? (
              <div className="p-8 text-center text-red-600 font-semibold">{error}</div>
            ) : impactedTestCases.length === 0 ? (
              <div className="p-8 text-center text-muted-foreground">No modified test cases found for this story.</div>
            ) : (
              <div className="space-y-6">
                {/* Summary Cards */}
                <div className="grid grid-cols-4 gap-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Total Test Cases</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{summaryData.total_test_cases}</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Modified Test Cases</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-amber-600">{summaryData.impacted_test_cases_count}</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Modification Percentage</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-blue-600">{summaryData.impact_percentage}%</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Severity Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-2 text-sm">
                        <span className="px-2 py-1 bg-red-100 text-red-700 rounded">High: {summaryData.severity_summary.high}</span>
                        <span className="px-2 py-1 bg-yellow-100 text-yellow-700 rounded">Med: {summaryData.severity_summary.medium}</span>
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded">Low: {summaryData.severity_summary.low}</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Test Cases Grid - Only showing modified test cases */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {impactedTestCases.map((testCase, idx) => (
                  <Card 
                    key={testCase.id || idx}
                    className="border border-border hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => handleViewDetails(testCase)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center font-semibold flex-shrink-0 ${
                            testCase.impacts[0]?.impact_severity === 'high' ? 'bg-red-100 text-red-800' :
                            testCase.impacts[0]?.impact_severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                          {idx + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                            {/* Modified Title */}
                          <div className="font-medium text-foreground text-base break-words mb-2">
                              {testCase.modified_title}
                          </div>
                            {/* Modified Test Case ID */}
                            <div className="text-sm text-muted-foreground mb-2">
                              <span className="font-medium">Modified ID: </span>
                            <span className="font-mono bg-muted/50 px-2 py-0.5 rounded">
                                {`${testCase.id}-MOD-${testCase.impacts.length}`}
                              </span>
                            </div>
                            {/* Impact Summary */}
                            <div className="flex items-center gap-2">
                              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                testCase.impacts[0]?.impact_severity === 'high' ? 'bg-red-100 text-red-700' :
                                testCase.impacts[0]?.impact_severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                'bg-green-100 text-green-700'
                              }`}>
                                Severity: {testCase.impacts[0]?.impact_severity || 'low'}
                              </span>
                              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                testCase.impacts[0]?.impact_priority === 'high' ? 'bg-purple-100 text-purple-700' :
                                testCase.impacts[0]?.impact_priority === 'medium' ? 'bg-blue-100 text-blue-700' :
                                'bg-gray-100 text-gray-700'
                              }`}>
                                Priority: {testCase.impacts[0]?.impact_priority || 'low'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
                </div>
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

      {/* Full Content Dialog */}
      <Dialog open={showFullContent} onOpenChange={setShowFullContent}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold text-purple-600 flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Page {selectedTestCase?.pageNumber} Content
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="bg-purple-50 rounded-lg p-6 border border-purple-100">
              <p className="text-foreground whitespace-pre-wrap">
                {selectedTestCase?.fullPageContent}
              </p>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Test Case Details Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold text-blue-600 flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Test Case Details
            </DialogTitle>
            <DialogDescription className="text-sm text-muted-foreground font-mono">
              Original ID: {selectedTestCase?.id} | Modified ID: {selectedTestCase ? `${selectedTestCase.id}-MOD-${selectedTestCase.impacts.length}` : 'N/A'}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            {/* Modified Title */}
            <div className="space-y-2">
              <div className="text-sm font-medium text-blue-600">Modified Title</div>
              <p className="text-base bg-blue-50 p-3 rounded-lg">
                {selectedTestCase?.modified_title}
              </p>
            </div>

            {/* Impact Reason */}
            <div className="space-y-2">
              <div className="text-sm font-medium text-amber-600">Impact Reason</div>
              <p className="text-base bg-amber-50 p-3 rounded-lg">
                {selectedTestCase?.impacts[0]?.impact_analysis_json ? 
                  JSON.parse(selectedTestCase.impacts[0].impact_analysis_json)?.modification_reason || 'No reason provided' 
                  : 'No reason provided'}
              </p>
            </div>

            {/* Modified Test Case Details */}
            <div className="space-y-4 border border-dashed border-emerald-200 bg-gradient-to-br from-emerald-50/50 to-blue-50/50 p-4 rounded-lg">
              <div className="flex items-center gap-2 text-emerald-600">
                <FileText className="h-4 w-4" />
                <h3 className="font-medium">Modified Test Case Details</h3>
              </div>

              {/* Modified Steps */}
              <div className="space-y-2">
                <div className="text-sm font-medium text-emerald-600">Modified Test Steps</div>
                <div className="space-y-2">
                  {selectedTestCase?.impacts[0]?.impact_analysis_json ? 
                    JSON.parse(selectedTestCase.impacts[0].impact_analysis_json)?.modified_test_case?.steps?.map((step: string, index: number) => (
                      <div key={index} className="flex items-start gap-3 bg-white p-3 rounded-lg border border-emerald-100">
                        <div className="flex-shrink-0 w-6 h-6 bg-emerald-100 rounded-full flex items-center justify-center text-emerald-700 text-sm font-medium">
                          {index + 1}
                        </div>
                        <div className="text-sm">{step}</div>
                      </div>
                    )) : 
                    <div className="text-sm text-muted-foreground italic">No modified steps available</div>
                  }
                </div>
              </div>

              {/* Modified Expected Result */}
              <div className="space-y-2">
                <div className="text-sm font-medium text-emerald-600">Modified Expected Result</div>
                <div className="text-sm leading-relaxed bg-white p-4 rounded-lg border border-emerald-100">
                  {selectedTestCase?.impacts[0]?.impact_analysis_json ? 
                    JSON.parse(selectedTestCase.impacts[0].impact_analysis_json)?.modified_test_case?.expected_result || 'No modified expected result available'
                    : 'No modified expected result available'}
                </div>
              </div>
            </div>

            {/* Story References Grid */}
            <div className="space-y-4 border border-dashed border-gray-200 bg-gradient-to-br from-gray-50/50 to-blue-50/50 p-4 rounded-lg">
              <div className="flex items-center gap-2 text-gray-600">
                <FileText className="h-4 w-4" />
                <h3 className="font-medium">Story & Test Case References</h3>
              </div>

              {/* IDs Section */}
              <div className="grid grid-cols-2 gap-4">
                {/* Left Column */}
                <div className="space-y-2">
                  <div>
                    <div className="text-sm font-medium text-purple-600">Original Story ID</div>
                    <div className="font-mono bg-purple-50 p-2 rounded-lg mt-1 border border-purple-100">
                      {storyId}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-purple-600">Original Test Case ID</div>
                    <div className="font-mono bg-purple-50 p-2 rounded-lg mt-1 border border-purple-100">
                      {selectedTestCase?.id || 'N/A'}
                    </div>
                  </div>
                </div>
                {/* Right Column */}
                <div className="space-y-2">
                  <div>
                    <div className="text-sm font-medium text-emerald-600">New Story ID</div>
                    <div className="font-mono bg-emerald-50 p-2 rounded-lg mt-1 border border-emerald-100">
                      {selectedTestCase?.impacts[0]?.impacting_story_id || 'N/A'}
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    className="w-full mt-1 group relative flex items-center justify-center gap-2 overflow-hidden bg-white transition-all hover:bg-blue-50 border-blue-200 text-blue-600 hover:border-blue-300"
                    onClick={() => setShowOriginalTestCase(prev => !prev)}
                  >
                    <Eye className="h-4 w-4" />
                    {showOriginalTestCase ? 'Hide Original' : 'View Original'}
                  </Button>
                </div>
              </div>

              {/* Priority Section */}
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <div className="text-sm font-medium text-purple-600">Original Priority</div>
                  <div className="flex items-center gap-2 bg-purple-50 p-2 rounded-lg mt-1 border border-purple-100">
                    <AlertCircle className="h-4 w-4 text-purple-600" />
                    <span className="font-medium">{selectedTestCase?.original_priority}</span>
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-emerald-600">Modified Priority</div>
                  <div className="flex items-center gap-2 bg-emerald-50 p-2 rounded-lg mt-1 border border-emerald-100">
                    <AlertCircle className="h-4 w-4 text-emerald-600" />
                    <span className="font-medium">
                      {selectedTestCase?.impacts[0]?.impact_analysis_json ? 
                        JSON.parse(selectedTestCase.impacts[0].impact_analysis_json)?.modified_test_case?.priority || 'Medium'
                        : 'Medium'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Original Test Case Section */}
              {showOriginalTestCase && (
                <div className="space-y-4 border border-dashed border-blue-200 bg-gradient-to-br from-blue-50/50 to-purple-50/50 p-4 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2 text-blue-600">
                      <FileText className="h-4 w-4" />
                      <h3 className="font-medium">Original Test Case</h3>
                    </div>
                  </div>

                  {loadingOriginal ? (
                    <div className="text-sm text-blue-600 text-center py-4">
                      Loading original test case...
                    </div>
                  ) : (
                    <>
                    {/* Original Title */}
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-purple-600">Original Title</div>
                      <div className="text-sm leading-relaxed bg-white p-4 rounded-lg border border-purple-100">
                        {selectedTestCase?.original_title || 'No title available'}
                      </div>
                    </div>

                      {/* Original Steps */}
                        <div className="space-y-2">
                          <div className="text-sm font-medium text-purple-600">Test Steps</div>
                          <div className="space-y-2">
                        {selectedTestCase?.original_steps?.map((step, index) => (
                              <div key={index} className="flex items-start gap-3 bg-white p-3 rounded-lg border border-purple-100">
                                <div className="flex-shrink-0 w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center text-purple-700 text-sm font-medium">
                                  {index + 1}
                                </div>
                                <div className="text-sm">{step}</div>
                              </div>
                            ))}
                          </div>
                        </div>

                      {/* Original Expected Result */}
                        <div className="space-y-2 mt-4">
                          <div className="text-sm font-medium text-purple-600">Expected Result</div>
                          <div className="text-sm leading-relaxed bg-white p-4 rounded-lg border border-purple-100">
                        {selectedTestCase?.original_expected_result || 'No expected result available'}
                      </div>
                    </div>
                  </>
            )}
              </div>
            )}

            {/* Impact Analysis by AI */}
            <div className="flex items-center gap-4 py-3 px-4 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex items-center gap-1.5 text-sm">
                <span className="font-medium text-red-600">Severity:</span>
                <span className="text-red-700">{selectedTestCase?.impacts[0]?.impact_severity?.toUpperCase() || 'LOW'}</span>
              </div>
              <div className="w-px h-4 bg-gray-300" />
              <div className="flex items-center gap-1.5 text-sm">
                <span className="font-medium text-orange-600">Priority:</span>
                <span className="text-orange-700">{selectedTestCase?.impacts[0]?.impact_priority?.toUpperCase() || 'LOW'}</span>
              </div>
              <div className="w-px h-4 bg-gray-300" />
              <div className="flex items-center gap-1.5 text-sm">
                <span className="font-medium text-violet-600">Last Modified:</span>
                <span className="text-violet-700">
                  {selectedTestCase?.impacts[0]?.impact_created_on ? 
                    new Date(selectedTestCase.impacts[0].impact_created_on).toLocaleString() : 
                    'N/A'
                  }
                </span>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

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
  let cleaned = response.trim();
  if (cleaned.startsWith('```json')) cleaned = cleaned.slice(7);
  if (cleaned.startsWith('```')) cleaned = cleaned.slice(3);
  if (cleaned.endsWith('```')) cleaned = cleaned.slice(0, -3);
  cleaned = cleaned.trim();
  try {
    const parsed = JSON.parse(cleaned);
    return { parsed, cleaned };
  } catch {
    return { parsed: null, cleaned };
  }
}

function formatRagTestCases(testCases: any[]) {
  return testCases.map((tc, idx) => {
    const steps = Array.isArray(tc.steps) ? tc.steps.join('\n') : tc.steps;
    return `Test Case ${idx + 1}: ${tc.title}\nSteps:\n${steps}\nExpected Result: ${tc.expected_result}\n`;
  }).join('\n\n');
}