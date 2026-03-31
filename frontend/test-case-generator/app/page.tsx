"use client"

import { useState, useEffect, useRef } from "react"
import {
  Download,
  Search,
  Eye,
  Triangle,
  AlertCircle,
  Linkedin,
  Facebook,
  Twitter,
  Instagram,
  Youtube,
  Music,
  ChevronLeft,
  ChevronRight,
  Clock,
  ArrowRight,
  FileText,
  RefreshCw,
  Calendar,
  ChevronUp,
  ChevronDown,
  Brain,
} from "lucide-react"
import { Chatbot } from "@/components/chatbot"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useRouter } from "next/navigation"
import { api, Story } from "@/lib/api"
import { toast } from "sonner"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface FormData {
  storyId: string
  storyDescription: string
  createdDate: string
  fromDate: string
  toDate: string
  priority?: string
  category?: string
  assignee?: string
}

interface TestCase {
  storyId: string
  storyDescription: string
  testCaseCount: number
  processStartTime: string
  processEndTime: string
  testCaseCreatedTime: string
  status?: "completed" | "in-progress" | "pending" | "failed"
  priority?: "high" | "medium" | "low"
  category?: string
  coverage?: number
  assignee?: string
}

interface DashboardStats {
  totalTestCases: number
  completedTestCases: number
  pendingTestCases: number
  failedTestCases: number
  averageCoverage: number
  totalStories: number
}

function calculateDuration(embeddingTime: string | null, testCaseTime: string | null): string {
  if (!embeddingTime || !testCaseTime) return 'N/A';
  
  const embedding = new Date(embeddingTime);
  const testCase = new Date(testCaseTime);
  const diffMs = testCase.getTime() - embedding.getTime();
  
  // Convert to minutes and seconds
  const minutes = Math.floor(diffMs / 60000);
  const seconds = Math.floor((diffMs % 60000) / 1000);
  
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function NextReloadBanner() {
  const [nextReload, setNextReload] = useState<Date | null>(null);
  const [timeLeft, setTimeLeft] = useState<string>("");
  const [isTriggering, setIsTriggering] = useState(false);

  const fetchNextReload = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5001/api/scheduler/next-reload");
      // Backend may return either plain ISO text or JSON: { next_reload: ISOString }
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const data = await response.json();
        const value = data?.next_reload ?? data?.nextReload ?? null;
        const date = value ? new Date(String(value).trim()) : null;
        if (date && !isNaN(date.getTime())) setNextReload(date);
      } else {
        const text = await response.text();
        const date = new Date(text.trim());
        if (!isNaN(date.getTime())) setNextReload(date);
      }
    } catch (error) {
      console.error("Error fetching next reload time:", error);
    }
  };

  const triggerReload = async () => {
    try {
      setIsTriggering(true);
      const response = await fetch("http://127.0.0.1:5001/api/scheduler/trigger", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        toast.success('Scheduler triggered successfully!');
        setTimeout(fetchNextReload, 2000);
      } else {
        toast.error('Failed to trigger scheduler');
      }
    } catch (error) {
      console.error("Error triggering reload:", error);
      toast.error('Failed to trigger scheduler');
    } finally {
      setIsTriggering(false);
    }
  };

  useEffect(() => {
    fetchNextReload();
    const pollInterval = setInterval(fetchNextReload, 30000);
    return () => clearInterval(pollInterval);
  }, []);

  useEffect(() => {
    if (!nextReload) return;
    
    const updateCountdown = () => {
      const now = new Date();
      const diff = Math.max(0, nextReload.getTime() - now.getTime());
      
      if (diff <= 0) {
        setTimeLeft("Reloading...");
        return;
      }
      
      const hours = String(Math.floor(diff / 3600000)).padStart(2, '0');
      const minutes = String(Math.floor((diff % 3600000) / 60000)).padStart(2, '0');
      const seconds = String(Math.floor((diff % 60000) / 1000)).padStart(2, '0');
      setTimeLeft(`${hours}:${minutes}:${seconds}`);
    };

    updateCountdown();
    const countdownInterval = setInterval(updateCountdown, 1000);
    return () => clearInterval(countdownInterval);
  }, [nextReload]);

  return (
    <div className="flex items-center gap-4 ml-auto bg-white px-4 py-2 rounded-xl shadow-sm border border-gray-200">
      <span className="text-sm font-medium text-gray-600">Next Scheduler Runs At:</span>
      <div className="flex items-center gap-2">
        <Clock className="w-4 h-4 text-blue-500" />
        <span className="font-mono text-sm font-semibold text-gray-900 min-w-[85px]">
          {timeLeft || "Loading..."}
        </span>
      </div>
      <Button
        onClick={triggerReload}
        disabled={isTriggering}
        className="h-9 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-lg font-semibold transition-all duration-200 flex items-center gap-2 shadow-md hover:shadow-lg transform hover:scale-105"
      >
        {isTriggering ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            <span>Triggering...</span>
          </>
        ) : (
          <>
            <RefreshCw className="w-4 h-4" />
            <span>Trigger</span>
          </>
        )}
      </Button>
    </div>
  );
}

function formatRagTestCases(testCases: any[]) {
  return testCases.map((tc, idx) => {
    return [
      `Test Case ${idx + 1}: ${tc.title}`,
      `Description: ${tc.description}`,
      `Steps:`,
      ...tc.steps.map((step: string, i: number) => `  ${i + 1}. ${step}`),
      `Expected Result: ${tc.expected_result}`,
      ''
    ].join('\n');
  }).join('\n');
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

export default function TestCaseGenerator() {
  const [formData, setFormData] = useState<FormData>({
    storyId: "",
    storyDescription: "",
    createdDate: "",
    fromDate: "",
    toDate: "",
  })

  const [showModal, setShowModal] = useState(false)
  const [selectedTestCases, setSelectedTestCases] = useState<string[]>([])
  const [searchPerformed, setSearchPerformed] = useState(false)
  const [activeTab, setActiveTab] = useState("dashboard")
  const [isLoading, setIsLoading] = useState(false)
  const [currentTime, setCurrentTime] = useState<string>('')
  const [chatMessages, setChatMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([
    {
      role: "assistant",
      content:
        "Hello! I'm your AI Test Case Assistant. I can help you in two ways:\n\n🔍 **RAG Mode (Default)**: Generate comprehensive test cases based on your user stories and existing test case database.\n\n💬 **Gemini Mode**: Ask questions about software testing, QA processes, and best practices.\n\nWhat would you like to do today?",
    },
  ])
  const [isGenerating, setIsGenerating] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [stories, setStories] = useState<Story[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [loading, setLoading] = useState(true)
  const [formattedDates, setFormattedDates] = useState<Record<string, string>>({})
  const router = useRouter()
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [perPage, setPerPage] = useState(10)
  const [totalStories, setTotalStories] = useState(0)
  const [dateFilter, setDateFilter] = useState<{from: string, to: string}>({from: '', to: ''});
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [selectedStory, setSelectedStory] = useState<Story | null>(null);
  const [showStoryContent, setShowStoryContent] = useState(false);
  const [lastReloadTime, setLastReloadTime] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<'gemini' | 'rag'>('rag');
  const [projects, setProjects] = useState<string[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');
  const [uploadForm, setUploadForm] = useState({
    projectId: '',
    storyId: '',
    content: '',
    file: null as File | null
  });
  const [newProjectName, setNewProjectName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: 'success' | 'error' | 'info';
    message: string;
  } | null>(null);
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [showContentDialog, setShowContentDialog] = useState(false);

  // Auto-remove success messages after 5 seconds
  useEffect(() => {
    if (uploadStatus?.type === 'success') {
      const timer = setTimeout(() => {
        setUploadStatus(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [uploadStatus]);

  // Update time on mount and every second
  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString());
    };
    
    // Initial update
    updateTime();
    
    // Update every second
    const interval = setInterval(updateTime, 1000);
    
    // Cleanup
    return () => clearInterval(interval);
  }, []);

  // Fetch paginated stories from backend
  const fetchStories = async (page = 1, perPage = 10) => {
    try {
      setLoading(true);
      const response = await api.getStories(page, perPage, dateFilter.from, dateFilter.to, selectedProject, sortOrder);
      
      if (response.stories && Array.isArray(response.stories)) {
        setStories(response.stories);
        setTotalPages(Math.ceil(response.pagination.total / response.pagination.per_page));
        setTotalStories(response.pagination.total);
        setCurrentPage(response.pagination.page);
      } else {
        console.error('Invalid response format:', response);
        toast.error('Failed to fetch stories: Invalid response format');
      }
    } catch (error) {
      console.error('Error fetching stories:', error);
      toast.error('Failed to fetch stories');
    } finally {
      setLoading(false);
    }
  };

  const fetchProjects = async () => {
    try {
      const response = await api.getProjects();
      setProjects(response.projects);
    } catch (error) {
      console.error('Error fetching projects:', error);
    }
  };

  useEffect(() => {
    fetchStories(currentPage, perPage);
  }, [currentPage, dateFilter, selectedProject, sortOrder]);

  useEffect(() => {
    fetchProjects();
  }, []);

  // Debug: log stories and their timestamps
  useEffect(() => {
    console.debug('Fetched stories:', stories);
    stories.forEach((story, idx) => {
      console.debug(`Story[${idx}] id=${story.id} embedding_timestamp=`, story.embedding_timestamp, 'test_case_created_time=', story.test_case_created_time);
    });
  }, [stories]);

  // Format dates after component mounts
  useEffect(() => {
    const dates: Record<string, string> = {};
    stories.forEach(story => {
      if (story.created_on) {
        dates[story.id] = new Date(story.created_on).toLocaleString();
      }
    });
    setFormattedDates(dates);
  }, [stories]);

  // Search stories with advanced filtering
  const handleSearch = async () => {
    setLoading(true);

    // If ID is entered, filter by ID (exact match)
    if (formData.storyId.trim()) {
      let filtered = [...stories].filter((story: any) => story.id.toLowerCase() === formData.storyId.trim().toLowerCase());
      // Further filter by date range if set
      if (formData.fromDate || formData.toDate) {
        filtered = filtered.filter((story: any) => {
          if (!story.created_on) return false;
          const created = new Date(story.created_on);
          const from = formData.fromDate ? new Date(formData.fromDate) : null;
          const to = formData.toDate ? new Date(formData.toDate) : null;
          if (from && created < from) return false;
          if (to && created > to) return false;
          return true;
        });
      }
      setStories(filtered);
      if (filtered.length === 0) {
        toast.info('No stories found with the specified ID');
      }
      setLoading(false);
      return;
    }

    // If description is entered, use similarity search
    if (formData.storyDescription.trim()) {
      try {
        const data = await api.searchStories(formData.storyDescription.trim(), 3); // fetch up to 3 similar stories
        let filtered = data.stories || [];
        
        if (filtered.length === 0) {
          setStories([]);
          toast.info('No matching stories found');
          setLoading(false);
          return;
        }
        
        // Further filter by date range if set
        if (formData.fromDate || formData.toDate) {
          filtered = filtered.filter((story: any) => {
            if (!story.created_on) return false;
            const created = new Date(story.created_on);
            const from = formData.fromDate ? new Date(formData.fromDate) : null;
            const to = formData.toDate ? new Date(formData.toDate) : null;
            if (from && created < from) return false;
            if (to && created > to) return false;
            return true;
          });
        }
        setStories(filtered);
        if (filtered.length === 0) {
          toast.info('No stories found matching the description and date range');
        } else {
          toast.success(`Found ${filtered.length} matching story${filtered.length > 1 ? 'ies' : ''}`);
        }
      } catch (e) {
        setStories([]);
        toast.error('Failed to perform similarity search.');
      }
      setLoading(false);
      return;
    }

    // If only date range is set, filter all stories by date
    if (formData.fromDate || formData.toDate) {
      let filtered = [...stories].filter((story: any) => {
        if (!story.created_on) return false;
        const created = new Date(story.created_on);
        const from = formData.fromDate ? new Date(formData.fromDate) : null;
        const to = formData.toDate ? new Date(formData.toDate) : null;
        if (from && created < from) return false;
        if (to && created > to) return false;
        return true;
      });
      setStories(filtered);
      if (filtered.length === 0) {
        toast.info('No stories found in the specified date range');
      }
      setLoading(false);
      return;
    }

    // If nothing is entered, fetch all stories
    fetchStories(currentPage, perPage);
    setLoading(false);
  };

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const openTestCasesModal = (count: number) => {
    const steps = []
    for (let i = 1; i <= count; i++) {
      steps.push(`Step ${i}: ${getRandomTestStep()}`)
    }
    setSelectedTestCases(steps)
    setShowModal(true)
  }

  const getRandomTestStep = () => {
    const steps = [
      "Open login page",
      "Enter valid credentials",
      "Click submit button",
      "Verify successful login",
      "Navigate to dashboard",
      "Check user profile",
      "Logout from application",
      "Verify error messages",
      "Test input validation",
      "Check responsive design",
    ]
    return steps[Math.floor(Math.random() * steps.length)]
  }

  const getUserSearchResult = (): TestCase => ({
    storyId: formData.storyId || "USER-001",
    storyDescription: formData.storyDescription || "User entered test case",
    testCaseCount: Math.floor(Math.random() * 10) + 1,
    processStartTime: new Date().toLocaleString(),
    processEndTime: new Date(Date.now() + 15 * 60000).toLocaleString(),
    testCaseCreatedTime: new Date(Date.now() + 20 * 60000).toLocaleString(),
  })

  const handleSendMessage = async (message: string) => {
    if (!message.trim() || isGenerating) return

    setApiError(null)

    // Add user message to chat
    setChatMessages((prev) => [...prev, { role: "user", content: message }])
    setIsGenerating(true)

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
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to generate response")
      }

      if (data.error) {
        setApiError(data.error)
      }

      // If RAG, display structured test cases
      if (selectedModel === 'rag' && data.response) {
        const { parsed, cleaned } = cleanAndParseLLMResponse(data.response);
        console.log('RAG backend data.response:', data.response);
        console.log('RAG cleaned:', cleaned);
        console.log('RAG parsed:', parsed);
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
        setChatMessages((prev) => [...prev, { role: "assistant", content: formatRagTestCases(data.testCases) }])
      } else {
        // For Gemini mode, display the response directly
        setChatMessages((prev) => [...prev, { role: "assistant", content: data.response }])
      }
    } catch (error) {
      console.error("Error generating response:", error)
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: selectedModel === 'rag' 
            ? "I'm using mock test cases for demonstration. In production, you would connect to the RAG backend."
            : "I'm using mock QA support for demonstration. In production, you would connect to the Gemini API.",
        },
      ])
      setApiError(selectedModel === 'rag' 
        ? "Using mock test cases. To use RAG backend, ensure the Flask server is running."
        : "Using mock QA support. To use Gemini API, add GOOGLE_GENERATIVE_AI_API_KEY to your environment.")
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownload = async (storyId: string) => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/stories/testcases/download/${storyId}`);
      
      if (!response.ok) {
        throw new Error('Failed to download test cases');
      }

      // Get the blob from the response
      const blob = await response.blob();
      
      // Create a URL for the blob
      const url = window.URL.createObjectURL(blob);
      
      // Create a temporary link element
      const link = document.createElement('a');
      link.href = url;
      link.download = `test_cases_${storyId}.xlsx`;
      
      // Append to body, click, and remove
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the URL
      window.URL.revokeObjectURL(url);
      
      toast.success('Test cases downloaded successfully');
    } catch (error) {
      console.error('Error downloading test cases:', error);
      toast.error('Failed to download test cases');
    }
  };

  const handleDateFilter = async () => {
    setLoading(true);
    try {
      if (formData.storyDescription.trim()) {
        // Similarity search, then filter by date
        const data = await api.searchStories(formData.storyDescription.trim(), 3);
        let filtered = data.stories || [];
        if (dateFilter.from || dateFilter.to) {
          filtered = filtered.filter((story: any) => {
            if (!story.created_on) return false;
            const created = new Date(story.created_on);
            const from = dateFilter.from ? new Date(dateFilter.from) : null;
            const to = dateFilter.to ? new Date(dateFilter.to) : null;
            if (from && created < from) return false;
            if (to && created > to) return false;
            return true;
          });
        }
        setStories(filtered);
        setTotalPages(1);
        setCurrentPage(1);
      } else {
        // Paginated endpoint with date filter
        const data = await api.getStories(currentPage, perPage, dateFilter.from, dateFilter.to);
        setStories(data.stories || []);
        setTotalPages(data.total_pages || 1);
        setCurrentPage(data.current_page || 1);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClearFilters = async () => {
    setDateFilter({from: '', to: ''});
    setSelectedProject(''); // Clear project filter
    setFormData(prev => ({ ...prev, storyId: '', storyDescription: '' }));
    setLoading(true);
    try {
      const data = await api.getStories(currentPage, perPage);
      setStories(data.stories || []);
      setTotalPages(data.total_pages || 1);
      setCurrentPage(data.current_page || 1);
    } finally {
      setLoading(false);
    }
  };

  // Pagination controls
  const renderPagination = () => {
    if (totalPages <= 0) return null;

    // Calculate the range of page numbers to display
    const pageRange = 5; // Number of page buttons to show
    let startPage = Math.max(1, currentPage - Math.floor(pageRange / 2));
    let endPage = Math.min(totalPages, startPage + pageRange - 1);

    // Adjust start page if we're near the end
    if (endPage - startPage + 1 < pageRange) {
      startPage = Math.max(1, endPage - pageRange + 1);
    }

    // Generate array of page numbers to display
    const pages = Array.from({ length: endPage - startPage + 1 }, (_, i) => startPage + i);

    const handlePageChange = async (newPage: number) => {
      if (newPage !== currentPage && newPage >= 1 && newPage <= totalPages) {
        await fetchStories(newPage, perPage);
      }
    };

    const handlePerPageChange = async (newPerPage: number) => {
      setPerPage(newPerPage);
      await fetchStories(1, newPerPage);
    };

    return (
      <div className="flex items-center justify-between px-4 py-4 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-700">
            Showing {((currentPage - 1) * perPage) + 1} to {Math.min(currentPage * perPage, totalStories)} of {totalStories} stories
          </span>
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Per page:</label>
            <select
              value={perPage}
              onChange={(e) => handlePerPageChange(Number(e.target.value))}
              className="border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-200"
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="px-3 py-2 border border-gray-300 hover:border-gray-400 hover:bg-gray-50 rounded font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>
          
          <div className="flex items-center gap-1">
            {/* First page button */}
            {startPage > 1 && (
              <>
                <Button
                  variant="outline"
                  onClick={() => handlePageChange(1)}
                  className="px-3 py-2 border border-gray-300 hover:border-gray-400 hover:bg-gray-50 rounded font-medium transition-colors text-sm"
                >
                  1
                </Button>
                {startPage > 2 && <span className="px-2">...</span>}
              </>
            )}

            {/* Page number buttons */}
            {pages.map((pageNum) => (
              <Button
                key={pageNum}
                variant={currentPage === pageNum ? "default" : "outline"}
                onClick={() => handlePageChange(pageNum)}
                className={`px-3 py-2 rounded font-medium transition-colors text-sm ${
                  currentPage === pageNum 
                    ? 'bg-blue-600 text-white' 
                    : 'border border-gray-300 hover:border-gray-400 hover:bg-gray-50'
                }`}
              >
                {pageNum}
              </Button>
            ))}

            {/* Last page button */}
            {endPage < totalPages && (
              <>
                {endPage < totalPages - 1 && <span className="px-2">...</span>}
                <Button
                  variant="outline"
                  onClick={() => handlePageChange(totalPages)}
                  className="px-3 py-2 border border-gray-300 hover:border-gray-400 hover:bg-gray-50 rounded font-medium transition-colors text-sm"
                >
                  {totalPages}
                </Button>
              </>
            )}
          </div>
          
          <Button
            variant="outline"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="px-3 py-2 border border-gray-300 hover:border-gray-400 hover:bg-gray-50 rounded font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  };


  const handleViewStoryContent = (e: React.MouseEvent, story: Story) => {
    e.stopPropagation();
    setSelectedStory(story);
    setShowStoryContent(true);
  };

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadForm(prev => ({ ...prev, file }));
    }
  };

  // Handle story upload
  const handleUploadStory = async () => {
    const projectId = uploadForm.projectId === 'new' ? newProjectName : uploadForm.projectId;
    
    if (!projectId || !uploadForm.storyId || (!uploadForm.content && !uploadForm.file)) {
      setUploadStatus({
        type: 'error',
        message: 'Please fill in Project, Story ID, and either type content or upload a document'
      });
      return;
    }

    setIsUploading(true);
    setUploadStatus({
      type: 'info',
      message: 'Uploading story and generating test cases...'
    });

    try {
      const formData = new FormData();
      formData.append('project_id', projectId);
      formData.append('story_id', uploadForm.storyId);
      formData.append('content', uploadForm.content);
      if (uploadForm.file) {
        formData.append('file', uploadForm.file);
      }

      const response = await fetch('/api/upload-story', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus({
          type: 'success',
          message: result.message + (result.description ? ` AI Description: "${result.description}"` : '')
        });
        
        // Reset form
        setUploadForm({
          projectId: '',
          storyId: '',
          content: '',
          file: null
        });
        setNewProjectName('');
        
        // Refresh stories list and projects
        fetchStories();
        fetchProjects();
      } else {
        setUploadStatus({
          type: 'error',
          message: result.error || 'Upload failed'
        });
      }
    } catch (error) {
      setUploadStatus({
        type: 'error',
        message: 'Network error. Please try again.'
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="fixed top-0 left-0 w-full z-50 bg-background shadow-sm border-b border-border px-4 md:px-8 py-4">
        <div className="flex justify-between items-center">
          <div className="flex-shrink-0">
            <img src="" alt="" className="h-12 w-auto" />
          </div>
          <div className="text-center">
            <h1 className="text-3xl font-bold text-primary">Test Case Generation</h1>
            <p className="text-sm text-muted-foreground font-semibold mt-1">Powered by Gen AI</p>
          </div>
          <div className="flex-shrink-0 flex items-center space-x-4">
            <div className="text-right">
              <div className="text-lg font-bold text-foreground">
                {currentTime || 'Loading...'}
              </div>
              <div className="text-sm text-muted-foreground">{new Date().toLocaleDateString()}</div>
            </div>
            
          </div>
        </div>
      </header>

      <div className="w-full py-6 pt-28 px-2">
        {/* Search Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              {showUploadForm ? 'Upload User Story' : 'Search User Stories'}
            </h2>
            <div className="flex items-center gap-2">
              <Button
                onClick={() => setShowUploadForm(!showUploadForm)}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl px-4 py-2 shadow-lg transform transition-all duration-200 hover:scale-105 flex items-center gap-2"
              >
                {showUploadForm ? (
                  <>
                    <Search className="w-4 h-4" />
                    Search Stories
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4" />
                    Upload Document
                  </>
                )}
              </Button>
            </div>
          </div>

          <Card className="shadow-xl border-0 bg-gradient-to-r from-white to-slate-50/50 overflow-hidden">
            <CardContent className="space-y-6 px-2 py-6 bg-white">
              {!showUploadForm ? (
                <>
                  <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        Story ID
                      </label>
                      <Input
                        placeholder="Enter Story ID"
                        value={formData.storyId}
                        onChange={(e) => handleInputChange("storyId", e.target.value)}
                        className="w-full border-2 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 h-10 rounded-xl shadow-sm"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        Story Description
                      </label>
                      <Textarea
                        placeholder="Enter story description..."
                        rows={2}
                        value={formData.storyDescription}
                        onChange={(e) => handleInputChange("storyDescription", e.target.value)}
                        className="w-full border-2 border-gray-200 focus:border-green-400 focus:ring-2 focus:ring-green-100 resize-none rounded-xl shadow-sm"
                      />
                    </div>
                  </div>

                  <div className="flex justify-end pt-4">
                    <Button
                      onClick={handleSearch}
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl px-8 py-3 shadow-lg transform transition-all duration-200 hover:scale-105 flex items-center gap-2"
                    >
                      <Search className="w-5 h-5" />
                      Search Stories
                    </Button>
                  </div>
                </>
              ) : (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        Project *
                      </label>
                      <select
                        value={uploadForm.projectId}
                        onChange={(e) => setUploadForm(prev => ({ ...prev, projectId: e.target.value }))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:border-blue-400 focus:ring-1 focus:ring-blue-200 transition-colors bg-white h-10"
                      >
                        <option value="">Select a project</option>
                        {projects.map((project) => (
                          <option key={project} value={project}>
                            {project}
                          </option>
                        ))}
                        <option value="new">+ Create New Project</option>
                      </select>
                      {uploadForm.projectId === 'new' && (
                        <Input
                          placeholder="Enter new project name"
                          value={newProjectName}
                          onChange={(e) => setNewProjectName(e.target.value)}
                          className="mt-2 w-full"
                        />
                      )}
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        Story ID *
                      </label>
                      <Input
                        placeholder="Enter Story ID"
                        value={uploadForm.storyId}
                        onChange={(e) => setUploadForm(prev => ({ ...prev, storyId: e.target.value }))}
                        className="w-full border border-gray-300 focus:border-blue-400 focus:ring-1 focus:ring-blue-200 h-10 rounded-lg"
                      />
                    </div>
                  </div>

                  <div className="space-y-2 md:col-span-4">
                    <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      User Story Content *
                    </label>
                    <div className="relative">
                      <div className="flex items-end gap-2">
                        <div className="flex-1 relative max-w-[1600px]">
                          <Textarea
                            placeholder="Type your user story here or upload a document..."
                            rows={3}
                            value={uploadForm.content}
                            onChange={(e) => setUploadForm(prev => ({ ...prev, content: e.target.value }))}
                            className="border border-gray-300 focus:border-blue-400 focus:ring-1 focus:ring-blue-200 resize-none rounded-lg pr-12 w-[1500px]"
                          />
                          <div className="absolute bottom-2 right-2 flex items-center gap-1">
                            <input
                              type="file"
                              accept=".pdf,.docx,.txt"
                              onChange={(e) => setUploadForm(prev => ({ ...prev, file: e.target.files?.[0] || null }))}
                              className="hidden"
                              id="file-upload-chat"
                            />
                            <label htmlFor="file-upload-chat" className="cursor-pointer p-1 hover:bg-gray-100 rounded transition-colors">
                              <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                              </svg>
                            </label>
                          </div>
                        </div>
                        <Button
                          onClick={handleUploadStory}
                          disabled={isUploading || 
                            (uploadForm.projectId === 'new' ? !newProjectName : !uploadForm.projectId) || 
                            !uploadForm.storyId || 
                            (!uploadForm.content && !uploadForm.file)}
                          className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-400 text-white font-semibold rounded-lg px-4 py-2 h-10 shadow-lg transform transition-all duration-200 hover:scale-105 whitespace-nowrap"
                        >
                          {isUploading ? (
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                              <span>Processing...</span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-2">
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                              </svg>
                              <span>Generate</span>
                            </div>
                          )}
                        </Button>
                      </div>
                      
                      {/* File Upload Status */}
                      {uploadForm.file && (
                        <div className="mt-2 flex items-center gap-2 p-3 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200">
                          <div className="w-6 h-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                          </div>
                          <span className="text-sm font-medium text-gray-800 flex-1">{uploadForm.file.name}</span>
                          <button
                            onClick={() => setUploadForm(prev => ({ ...prev, file: null }))}
                            className="text-gray-500 hover:text-gray-700 p-1 hover:bg-gray-100 rounded-lg transition-colors"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Upload Status */}
                  {uploadStatus && (
                    <div className={`p-4 rounded-xl border-2 ${
                      uploadStatus.type === 'success' ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-200 text-green-800' :
                      uploadStatus.type === 'error' ? 'bg-gradient-to-r from-red-50 to-pink-50 border-red-200 text-red-800' :
                      'bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200 text-blue-800'
                    }`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          {uploadStatus.type === 'success' && (
                            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            </div>
                          )}
                          {uploadStatus.type === 'error' && (
                            <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center">
                              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </div>
                          )}
                          {uploadStatus.type === 'info' && (
                            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            </div>
                          )}
                          <span className="font-semibold">{uploadStatus.message}</span>
                        </div>
                        <button
                          onClick={() => setUploadStatus(null)}
                          className="text-gray-500 hover:text-gray-700 p-1 hover:bg-gray-100 rounded-lg transition-colors"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Enhanced Date Filter Section */}
        <div className="mb-8">
          <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
            {/* Date filter controls - left side */}
            <div className="flex flex-col sm:flex-row sm:items-end gap-4">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-blue-500" />
                  From Date
                </label>
                <Input
                  type="date"
                  value={dateFilter.from}
                  onChange={e => setDateFilter(df => ({...df, from: e.target.value}))}
                  className="border-2 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 h-12 rounded-xl shadow-sm"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-green-500" />
                  To Date
                </label>
                <Input
                  type="date"
                  value={dateFilter.to}
                  onChange={e => setDateFilter(df => ({...df, to: e.target.value}))}
                  className="border-2 border-gray-200 focus:border-green-400 focus:ring-2 focus:ring-green-100 h-12 rounded-xl shadow-sm"
                />
              </div>
              <Button
                onClick={handleDateFilter}
                className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white font-semibold rounded-xl shadow-lg h-12 px-6 transition-all duration-200 transform hover:scale-105"
              >
                <Calendar className="mr-2 h-4 w-4" />
                Filter by Date
              </Button>
            </div>

            {/* Project filter and sort controls - right side */}
            <div className="flex flex-col lg:flex-row items-start lg:items-end gap-4">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                  Project
                </label>
                <select
                  value={selectedProject}
                  onChange={(e) => setSelectedProject(e.target.value)}
                  className="border-2 border-gray-200 focus:border-orange-400 focus:ring-2 focus:ring-orange-100 h-12 rounded-xl px-4 py-2 bg-white shadow-sm"
                >
                  <option value="">All Projects</option>
                  {projects.map((project) => (
                    <option key={project} value={project}>
                      {project}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  onClick={() => fetchStories(currentPage, perPage)}
                  variant="outline"
                  className="h-12 px-4 border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 rounded-xl font-semibold transition-all duration-200 flex items-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  Refresh
                </Button>
              </div>
              <div className="flex items-center gap-3">
                <Button
                  onClick={handleClearFilters}
                  variant="outline"
                  className="h-12 px-6 border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 rounded-xl font-semibold transition-all duration-200"
                >
                  Clear All Filters
                </Button>
                <NextReloadBanner />
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Results Table Section */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-800">Test Cases Results</h2>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 bg-white px-3 py-2 rounded-lg shadow-sm border border-gray-200">
                <label className="text-sm font-semibold text-gray-700">Sort by:</label>
                <select
                  value={sortOrder}
                  onChange={(e) => setSortOrder(e.target.value as 'desc' | 'asc')}
                  className="border-0 focus:ring-0 bg-transparent text-sm font-medium text-blue-600"
                >
                  <option value="desc">Newest First</option>
                  <option value="asc">Oldest First</option>
                </select>
              </div>
            </div>
          </div>
          
          <Card className="shadow-lg border border-gray-200 overflow-hidden">
            <CardContent className="p-0 bg-white">
              {loading ? (
                <div className="p-8 text-center">
                  <div className="inline-flex items-center gap-3 text-blue-600 font-semibold">
                    <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    Loading stories...
                  </div>
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-blue-100">
                  <table className="w-full bg-white">
                    <thead>
                      <tr className="bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 border-b border-blue-200">
                        <th className="font-semibold text-blue-900 py-3 px-3 min-w-[100px] max-w-[120px] text-left whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            <span>Story ID</span>
                          </div>
                        </th>
                        <th className="font-semibold text-blue-900 py-3 px-3 min-w-[180px] max-w-[200px] text-left">
                          <div className="flex flex-col items-start gap-1">
                            <span>Story Description</span>
                            <div className="inline-flex items-center gap-1 px-2 py-0.5 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                              <Brain className="w-3 h-3" />
                              AI Generated Description
                            </div>
                          </div>
                        </th>
                        <th className="font-semibold text-blue-900 py-3 px-3 min-w-[200px] max-w-[220px] text-left">
                          <div className="flex flex-col items-start gap-1">
                            <span>Story Content</span>
                            <div className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                              <FileText className="w-3 h-3" />
                              Content
                            </div>
                          </div>
                        </th>
                        <th className="font-semibold text-blue-900 py-3 px-3 min-w-[140px] max-w-[160px] text-center">
                          <div className="flex flex-col items-center gap-1">
                            <span>Test Cases</span>
                            <div className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                              <FileText className="w-3 h-3" />
                              Generated by AI
                            </div>
                          </div>
                        </th>
                        <th className="font-semibold text-blue-900 py-3 px-3 min-w-[140px] max-w-[160px] text-center">
                          <div className="flex flex-col items-center gap-1">
                            <span>Impacted Test Cases</span>
                            <div className="inline-flex items-center gap-1 px-2 py-0.5 bg-amber-100 text-amber-800 rounded-full text-xs font-medium">
                              <Brain className="w-3 h-3" />
                              Impact Analysis by AI
                            </div>
                          </div>
                        </th>
                        <th className="font-semibold text-blue-900 py-3 px-3 min-w-[140px] max-w-[160px] text-center">
                          <div className="flex flex-col items-center gap-1">
                            <span>Generated Time</span>
                            <div className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-100 text-gray-800 rounded-full text-xs font-medium">
                              <Calendar className="w-3 h-3" />
                              Date
                            </div>
                          </div>
                        </th>
                        <th className="font-semibold text-blue-900 py-3 px-3 min-w-[100px] max-w-[120px] text-center">
                          <div className="flex flex-col items-center gap-1">
                            <span>Actions</span>
                            <div className="inline-flex items-center gap-1 px-2 py-0.5 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                              <Download className="w-3 h-3" />
                              Download
                            </div>
                          </div>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {stories.map((story, index) => (
                        <tr key={story.id || index} className="group hover:bg-gray-50 transition-colors border-b border-gray-100">
                          <td className="font-semibold text-gray-800 py-2 px-3 min-w-[100px] max-w-[120px] relative group">
                            <div className="flex items-center gap-1.5">
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                              <span className="text-sm truncate cursor-default">{story.id}</span>
                              <div className="absolute left-3 top-full mt-1 z-10 invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                <div className="bg-white border border-gray-200 shadow-lg rounded-lg px-3 py-2 text-sm whitespace-nowrap">
                                  <div className="flex items-center gap-1.5">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                    {story.id}
                                  </div>
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="text-gray-700 py-2 px-3 min-w-[180px] max-w-[200px]">
                            <div className="relative group cursor-pointer">
                              <div className="description-content line-clamp-2 text-sm">
                                {story.description || 'No description available'}
                              </div>
                              {story.description && (
                                <button
                                  onClick={() => {
                                    setSelectedStory(story);
                                    setShowStoryContent(true);
                                  }}
                                  className="text-xs text-blue-600 hover:text-blue-700 font-medium mt-1"
                                >
                                  Read More
                                </button>
                              )}
                            </div>
                          </td>
                          <td className="text-gray-700 py-2 px-3 min-w-[200px] max-w-[220px]">
                            <div className="relative group cursor-pointer">
                              <div className="line-clamp-2 text-sm">
                                {story.doc_content_text || 'No content available'}
                              </div>
                              {story.doc_content_text && (
                                <button
                                  onClick={() => {
                                    setSelectedStory(story);
                                    setShowStoryContent(true);
                                  }}
                                  className="text-xs text-blue-600 hover:text-blue-700 font-medium mt-1"
                                >
                                  Read More
                                </button>
                              )}
                            </div>
                          </td>
                          <td className="py-2 px-3 text-center min-w-[140px] max-w-[160px]">
                            <button
                              onClick={() => router.push(`/test-cases/${story.id}`)}
                              className="inline-flex items-center justify-center px-2.5 py-1.5 rounded bg-blue-50 text-blue-700 font-medium hover:bg-blue-100 transition-colors border border-blue-200 text-sm"
                            >
                              <FileText className="w-3 h-3 mr-1" />
                              {story.test_case_count || 0} Test Cases
                            </button>
                          </td>
                          <td className="py-2 px-3 text-center min-w-[140px] max-w-[160px]">
                            {typeof story.impactedTestCases === 'number' ? (
                              <button
                                onClick={() => router.push(`/story-details/${story.id}?project_id=${story.project_id}`)}
                                className="inline-flex items-center justify-center px-2.5 py-1.5 rounded bg-amber-50 text-amber-700 font-medium hover:bg-amber-100 transition-colors border border-amber-200 text-sm"
                              >
                                <Brain className="w-3 h-3 mr-1" />
                                {story.impactedTestCases} Impacted
                              </button>
                            ) : (
                              <span className="text-gray-400 italic text-sm">N/A</span>
                            )}
                          </td>
                          <td className="py-2 px-3 text-center min-w-[140px] max-w-[160px]">
                            {story.test_case_created_time ? (
                              <div className="flex flex-col items-center text-sm space-y-1">
                                <span className="text-gray-800 font-medium">
                                  {new Date(story.test_case_created_time).toLocaleString('en-US', {
                                    year: 'numeric',
                                    month: 'short',
                                    day: 'numeric'
                                  })}
                                </span>
                                <span className="text-gray-500 bg-gray-100 px-2 py-0.5 rounded text-xs">
                                  {new Date(story.test_case_created_time).toLocaleString('en-US', {
                                    hour: '2-digit',
                                    minute: '2-digit',
                                    hour12: true
                                  })}
                                </span>
                              </div>
                            ) : (
                              <span className="text-gray-400 italic text-sm">-</span>
                            )}
                          </td>
                          <td className="py-2 px-3 text-center min-w-[100px] max-w-[120px]">
                            <button
                              onClick={() => handleDownload(story.id)}
                              className="inline-flex items-center justify-center px-2.5 py-1.5 rounded bg-green-50 text-green-700 font-medium hover:bg-green-100 transition-colors border border-green-200 text-sm"
                            >
                              <Download className="w-3 h-3 mr-1" />
                              Download
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {renderPagination()}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Simplified Footer */}
      <footer className="mt-6">
        <div className="bg-gray-800 py-3 text-center text-gray-300 text-sm">
          <div className="w-full">
            
          </div>
        </div>
      </footer>

      {/* Enhanced Modal Popup */}
      <Dialog open={showModal} onOpenChange={setShowModal}>
        <DialogContent className="max-w-3xl rounded-2xl border-0 shadow-2xl">
          <DialogHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-800 p-6 -m-6 mb-6 rounded-t-2xl border-b border-blue-100">
            <DialogTitle className="text-xl font-bold flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <FileText className="w-4 h-4 text-white" />
              </div>
              Test Case Steps
            </DialogTitle>
            <DialogDescription className="text-blue-600 mt-2">
              AI-generated test cases for your user story
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 max-h-96 overflow-y-auto px-2">
            {selectedTestCases.map((step, index) => (
              <div key={index} className="p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl border-l-4 border-blue-400 shadow-sm">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 mt-0.5">
                    {index + 1}
                  </div>
                  <p className="text-sm font-medium text-slate-700 leading-relaxed">{step}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="flex justify-end pt-6 border-t border-gray-100">
            <Button
              onClick={() => setShowModal(false)}
              className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white px-8 py-3 rounded-xl font-semibold shadow-lg transition-all duration-200 transform hover:scale-105"
            >
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Replace the old chatbot implementation with the new component */}
      <Chatbot
        onSendMessage={handleSendMessage}
        chatMessages={chatMessages}
        isGenerating={isGenerating}
        selectedModel={selectedModel}
        onModelChange={(model) => setSelectedModel(model)}
      />

      {/* Story Content Dialog */}
      <Dialog open={showStoryContent} onOpenChange={setShowStoryContent}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold text-blue-800 flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <FileText className="w-4 h-4 text-white" />
              </div>
              Story Details
            </DialogTitle>
            <DialogDescription className="text-blue-600 mt-2 font-medium">
              ID: {selectedStory?.id}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                <span>AI-Generated Description</span>
                <div className="px-2 py-0.5 bg-purple-100 text-purple-800 rounded-full text-xs">AI</div>
              </h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-700 text-sm">{selectedStory?.description}</p>
              </div>
            </div>
            
            {selectedStory?.doc_content_text && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Full Story Content</h3>
                <div className="bg-gray-50 rounded-lg p-4 max-h-[400px] overflow-y-auto border border-gray-100">
                  <p className="text-gray-700 text-sm whitespace-pre-wrap">{selectedStory.doc_content_text}</p>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Enhanced Story Content Dialog */}
      <Dialog open={showContentDialog} onOpenChange={setShowContentDialog}>
        <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto rounded-2xl border-0 shadow-2xl">
          <DialogHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 -m-6 mb-6 rounded-t-2xl border-b border-blue-100">
            <DialogTitle className="text-2xl font-bold text-blue-800 flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <FileText className="w-4 h-4 text-white" />
              </div>
              Story Content
            </DialogTitle>
            <DialogDescription className="text-blue-600 mt-2 font-medium">
              Story ID: {selectedStory?.id}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Story Description</h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-700">{selectedStory?.description}</p>
              </div>
            </div>
            
            {/* Story Content */}
            {selectedStory?.doc_content_text && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Full Content</h3>
                <div className="bg-gray-50 rounded-lg p-4 max-h-[400px] overflow-y-auto">
                  <p className="text-gray-700 whitespace-pre-wrap">{selectedStory.doc_content_text}</p>
                </div>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

