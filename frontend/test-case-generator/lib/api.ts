// API Base URLs with fallbacks
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";
const SCHEDULER_API_URL = process.env.NEXT_PUBLIC_SCHEDULER_API_URL || "http://127.0.0.1:5001";

// Helper function for API calls with error handling
const fetchWithErrorHandling = async (url: string, options: RequestInit = {}) => {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return response;
  } catch (error) {
    if (error instanceof TypeError && error.message === 'Failed to fetch') {
      throw new Error('Unable to connect to the server. Please check if the backend is running.');
    }
    throw error;
  }
};

export interface Story {
  id: string;
  description: string;
  doc_content_text: string | null;
  created_on: string;
  test_case_count: number;
  embedding_timestamp: string | null;
  test_case_created_time: string | null;
  project_id: string;
  title?: string;
  summary?: string;
  impactedTestCases?: number;
  source?: {
    story: string;
    test_cases: string;
  };
}

export interface TestCase {
  test_case_id?: string;
  id?: string;
  title: string;
  description?: string;
  steps?: string[];
  expected_result?: string;
  expected_results?: string[];
  priority?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  original_version?: TestCase; // For comparison
}

export interface ImpactedTestCase extends TestCase {
  similarity_score: number;
  original_test_case?: TestCase;
}

export interface TestCaseResponse {
  storyID: string;
  storyDescription: string | null;
  project_id?: string;
  testcases: TestCase[];
}

export interface ImpactAnalysisResponse {
  story_id: string;
  impacted_test_cases: ImpactedTestCase[];
  total_impacted: number;
}

export const api = {
  // Fetch paginated stories
  getStories: async (page: number = 1, perPage: number = 10, from_date?: string, to_date?: string, project_id?: string, sort_order: 'desc' | 'asc' = 'desc') => {
    try {
      let url = `${API_BASE_URL}/api/stories/?page=${page}&per_page=${perPage}&sort_order=${sort_order}`;
      if (from_date) url += `&from_date=${encodeURIComponent(from_date)}`;
      if (to_date) url += `&to_date=${encodeURIComponent(to_date)}`;
      if (project_id) url += `&project_id=${encodeURIComponent(project_id)}`;
      
      const response = await fetchWithErrorHandling(url);
      const data = await response.json();
      
      if (!data.stories || !Array.isArray(data.stories) || !data.pagination) {
        throw new Error("Invalid response format from server");
      }
      
      return data;
    } catch (error) {
      console.error("API Error:", error);
      throw error;
    }
  },

  // Search stories
  searchStories: async (query: string, limit: number = 3) => {
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/stories/search`, {
      method: "POST",
      body: JSON.stringify({ query, limit }),
    });
    return response.json();
  },

  // Fetch test cases for a story
  getTestCases: async (storyId: string) => {
    const url = `${API_BASE_URL}/api/stories/${storyId}/testcases`;
    console.debug("[api] getTestCases url", { url, storyId });
    const response = await fetchWithErrorHandling(url);
    return response.json();
  },

  // Get unique project IDs
  getProjects: async () => {
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/stories/projects`);
    return response.json();
  },

  // New methods for impact analysis
  getStoryImpacts: async (storyId: string): Promise<ImpactAnalysisResponse> => {
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/stories/impacts/story/${storyId}`);
    return response.json();
  },

  getImpactDetails: async (impactId: string) => {
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/stories/impacts/details/${impactId}`);
    return response.json();
  },

  getImpactSummary: async (projectId: string) => {
    const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/stories/impacts/summary/${projectId}`);
    return response.json();
  },

  fetchImpactedTestCases: async (storyId: string, projectId: string) => {
    try {
      const response = await fetchWithErrorHandling(
        `${API_BASE_URL}/api/stories/impacts/story-test-cases/${storyId}?project_id=${projectId}`
      );
      return response.json();
    } catch (error) {
      console.error('Error fetching impacted test cases:', error);
      throw error;
    }
  },

  // Scheduler endpoints
  getNextReload: async () => {
    const response = await fetchWithErrorHandling(`${SCHEDULER_API_URL}/api/scheduler/next-reload`);
    return response.text();
  },

  triggerReload: async () => {
    const response = await fetchWithErrorHandling(`${SCHEDULER_API_URL}/api/scheduler/trigger`, {
      method: 'POST',
    });
    return response.json();
  },
}; 