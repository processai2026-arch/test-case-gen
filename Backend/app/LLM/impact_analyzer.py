import json
from urllib import response
from urllib import response
import uuid
from datetime import datetime, timedelta
import lancedb
import numpy as np
from ..config import Config
from ..models.db_service import DatabaseService
from ..models.postgress_writer import get_test_case_json_by_story_id
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import logging
from typing import Dict, List, Optional
import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('impact_analysis.log')

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load the LLM prompt for impact analysis
with open("app/LLM/impact_analysis_prompt.txt", "r", encoding="utf-8") as f:
    IMPACT_INSTRUCTIONS = f.read()

# Constants for rate limiting and retries
MAX_RETRIES = 3
MAX_CONCURRENT_ANALYSES = 3
MIN_WAIT_BETWEEN_CALLS = 1  # seconds
MAX_STORIES_TO_ANALYZE = 5
API_TIMEOUT = 30  # seconds

class RateLimiter:
    def __init__(self, calls_per_minute: int = 50):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        
    def can_make_call(self) -> bool:
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        return len(self.calls) < self.calls_per_minute
        
    def record_call(self):
        self.calls.append(time.time())
        
    def wait_if_needed(self):
        while not self.can_make_call():
            time.sleep(1)
        self.record_call()

# Global rate limiter instance
rate_limiter = RateLimiter()

class ImpactAnalysisError(Exception):
    """Base class for Impact Analysis errors"""
    pass

class StoryNotFoundError(ImpactAnalysisError):
    """Raised when a story cannot be found"""
    pass

class TestCasesNotFoundError(ImpactAnalysisError):
    """Raised when test cases cannot be found"""
    pass

class LLMError(ImpactAnalysisError):
    """Raised when there's an error with LLM processing"""
    pass

class DatabaseError(ImpactAnalysisError):
    """Raised when there's a database error"""
    pass

def get_db_service() -> DatabaseService:
    """
    Get the singleton instance of DatabaseService with proper error handling
    """
    try:
        if DatabaseService._instance is None:
            DatabaseService(
            postgres_config=Config.postgres_config(),
            lance_db_path=Config.LANCE_DB_PATH
        )
        return DatabaseService._instance
    except Exception as e:
        logger.error(f"Failed to get database service: {str(e)}")
        raise DatabaseError(f"Database service error: {str(e)}")

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_cls=LLMError
)
def get_llm_analysis(prompt: str, llm_ref) -> Dict:
    """
    Get analysis from LLM with retry logic and error handling
    """
    try:
        rate_limiter.wait_if_needed()
        logger.debug("Making LLM API call")
        
        # Add explicit JSON formatting instructions
        structured_prompt = f"""
{prompt}

CRITICAL: Your response MUST be a valid JSON object with this exact structure:
{{
    "has_impact": boolean,
    "impact_type": "MODIFY" | "NO_IMPACT",
    "impacted_test_cases": [
        {{
            "original_test_case_id": "string",
            "modification_reason": "string",
            "modified_test_case": {{
                "id": "string",
                "title": "string",
                "steps": ["string"],
                "expected_result": "string",
                "priority": "High" | "Medium" | "Low"
            }}
        }}
    ]
}}

Do not include any explanatory text before or after the JSON.
Do not use markdown code blocks.
Just return the raw JSON object.
"""
        
        # Get response from LLM
        response = llm_ref.invoke(structured_prompt)

        if hasattr(response, "content"):
            content = str(response.content).strip()
        else:
            content = str(response).strip()
            
        logger.info(f"LLM Response: {content[:500]}")
        
        # Clean up the response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        # Try to find JSON object if there's any text before it
        if not content.startswith("{"):
            start_idx = content.find("{")
            if start_idx != -1:
                content = content[start_idx:]
                
        # Try to find JSON object if there's any text after it
        if not content.endswith("}"):
            end_idx = content.rfind("}") + 1
            if end_idx != 0:
                content = content[:end_idx]
        
        try:
            result = json.loads(content)

            # --- NORMALIZE COMMON LLM VARIANTS BEFORE STRICT VALIDATION ---
            # Some models might use "impacts" instead of "impacted_test_cases"
            if "impacted_test_cases" not in result and isinstance(result.get("impacts"), list):
                logger.info("Normalizing LLM field 'impacts' -> 'impacted_test_cases'")
                result["impacted_test_cases"] = result["impacts"]

            # If has_impact missing but we clearly have impacted_test_cases, infer it
            if "has_impact" not in result and result.get("impacted_test_cases"):
                logger.info("Inferring has_impact=True from non-empty impacted_test_cases")
                result["has_impact"] = True

            # Default impact_type if missing but we have impacts
            if "impact_type" not in result and result.get("impacted_test_cases"):
                logger.info("Defaulting impact_type='MODIFY' because impacts are present")
                result["impact_type"] = "MODIFY"

            # Validate required fields (tolerant to typing issues)
            raw_has_impact = result.get("has_impact", False)
            if isinstance(raw_has_impact, str):
                normalized = raw_has_impact.strip().lower()
                if normalized in ["true", "1", "yes", "y"]:
                    result["has_impact"] = True
                elif normalized in ["false", "0", "no", "n"]:
                    result["has_impact"] = False
                else:
                    raise ValueError("has_impact must be a boolean (or a boolean-like string)")
            elif not isinstance(raw_has_impact, bool):
                raise ValueError("has_impact must be a boolean")

            impact_type = result.get("impact_type", "NO_IMPACT")
            if isinstance(impact_type, str):
                impact_type_norm = impact_type.strip().upper()
            else:
                impact_type_norm = impact_type

            if impact_type_norm not in ["MODIFY", "NO_IMPACT"]:
                raise ValueError("impact_type must be either 'MODIFY' or 'NO_IMPACT'")
            result["impact_type"] = impact_type_norm

            impacted_list = result.get("impacted_test_cases", [])
            if not isinstance(impacted_list, list):
                raise ValueError("impacted_test_cases must be a list")
            # Normalize missing/nullable lists to []
            result["impacted_test_cases"] = impacted_list or []
                
            # If we have impacts, validate each test case (do not assume strict typing)
            if result["has_impact"] and result["impact_type"] == "MODIFY":
                for test_case in result["impacted_test_cases"]:
                    if not isinstance(test_case.get("original_test_case_id"), str):
                        raise ValueError("original_test_case_id must be a string")
                    if not isinstance(test_case.get("modification_reason"), str):
                        raise ValueError("modification_reason must be a string")
                    if not isinstance(test_case.get("modified_test_case"), dict):
                        raise ValueError("modified_test_case must be an object")

                    modified = test_case["modified_test_case"]

                    # Normalize priority/steps coming from LLM
                    if isinstance(modified.get("priority"), str):
                        pr_norm = modified["priority"].strip().lower()
                        pr_map = {"high": "High", "medium": "Medium", "low": "Low"}
                        if pr_norm in pr_map:
                            modified["priority"] = pr_map[pr_norm]
                    if isinstance(modified.get("steps"), str):
                        # If model returned a single string instead of a list, wrap it.
                        modified["steps"] = [modified["steps"]]

                    if not all(isinstance(modified.get(field), str) for field in ["id", "title", "expected_result"]):
                        raise ValueError("modified_test_case fields must be strings")
                    if not isinstance(modified.get("steps"), list):
                        raise ValueError("steps must be a list")
                    if modified.get("priority") not in ["High", "Medium", "Low"]:
                        raise ValueError("priority must be High, Medium, or Low")
            
            # Final normalization: ensure has_impact and impact_type are correct types
            result["has_impact"] = bool(result["has_impact"])
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON structure: {str(e)}")
            logger.error(f"Raw content: {content[:500]}")  # Log first 500 chars of content
            raise LLMError("Invalid JSON response from LLM")
        except ValueError as e:
            logger.error(f"Invalid response format: {str(e)}")
            logger.error(f"Parsed content: {json.dumps(result, indent=2)}")
            raise LLMError(f"Invalid response format: {str(e)}")
            
    except Exception as e:
        logger.error(f"LLM processing error: {str(e)}")
        raise LLMError(f"LLM processing error: {str(e)}")

def store_impact_analysis(
    impact_data: Dict,
    project_id: str,
    new_story_id: str,
    existing_story_id: str,
    similarity_score: float
) -> int:
    """
    Store impact analysis results in the database with enhanced tracking
    """
    try:
        db_service = get_db_service()
        conn = psycopg2.connect(**db_service.postgres_config)
        impacts_stored = 0
        
        with conn:
            with conn.cursor() as cur:
                # First get the run_id for the original story
                cur.execute("""
                    SELECT run_id FROM test_cases 
                    WHERE story_id = %s
                """, (existing_story_id,))
                result = cur.fetchone()
                if not result:
                    raise DatabaseError(f"Could not find run_id for story {existing_story_id}")
                original_run_id = result[0]
                
                # Get the current timestamp
                current_time = datetime.now()
                
                for impact in impact_data.get("impacted_test_cases", []):
                    # Get the original test case ID
                    original_test_case_id = impact["original_test_case_id"]
                    
                    # Find the next modification number for this test case
                    cur.execute("""
                        SELECT modified_test_case_id 
                        FROM test_case_impacts 
                        WHERE original_test_case_id = %s 
                        AND impact_status = 'active'
                        ORDER BY modified_test_case_id DESC
                        LIMIT 1
                    """, (original_test_case_id,))
                    last_mod = cur.fetchone()
                    
                    # Determine the next modification number
                    if last_mod and last_mod[0]:
                        last_mod_id = last_mod[0]
                        if "-mod-" in last_mod_id:
                            try:
                                mod_num = int(last_mod_id.split("-mod-")[1]) + 1
                            except:
                                mod_num = 1
                        else:
                            mod_num = 1
                    else:
                        mod_num = 1
                        
                    # Create the modified test case ID
                    modified_test_case_id = f"{original_test_case_id}-mod-{mod_num}"
                    
                    # Update the modified test case ID in the impact data
                    impact["modified_test_case"]["id"] = modified_test_case_id
                    
                    # Get severity directly from LLM analysis
                    severity = impact.get("impact_severity", "medium").lower()
                    severity_reason = impact.get("severity_reason", "No reason provided")
                    
                    # Determine impact priority (1-5 scale) based on severity
                    priority = {
                        "high": 5,
                        "medium": 3,
                        "low": 1
                    }.get(severity, 3)
                    
                    # Store impact details
                    impact_details = {
                        "modification_reason": impact["modification_reason"],
                        "severity_reason": severity_reason,
                        "changes": {
                            "title": impact["modified_test_case"]["title"],
                            "steps": impact["modified_test_case"]["steps"],
                            "expected_result": impact["modified_test_case"]["expected_result"]
                        }
                    }
                    
                    # Insert into test_case_impacts
                    cur.execute("""
                        INSERT INTO test_case_impacts (
                            impact_id,
                            project_id,
                            new_story_id,
                            original_story_id,
                            original_test_case_id,
                            modified_test_case_id,
                            original_run_id,
                            impact_created_on,
                            source,
                            similarity_score,
                            impact_analysis_json,
                            impact_status,
                            impact_type,
                            impact_severity,
                            impact_priority,
                            impact_details
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING impact_id
                    """, (
                        str(uuid.uuid4()),  # impact_id
                        project_id,
                        new_story_id,
                        existing_story_id,
                        original_test_case_id,
                        modified_test_case_id,
                        original_run_id,
                        current_time,
                        'llm',
                        similarity_score,
                        json.dumps(impact),
                        'active',  # Default status
                        'modification',  # Default type for now
                        severity,
                        priority,
                        json.dumps(impact_details)
                    ))

                    inserted_impact_id = cur.fetchone()[0]
                    # Keep impact_history linked for traceability/auditing
                    cur.execute("""
                        INSERT INTO impact_history (
                            history_id,
                            impact_id,
                            changed_by,
                            previous_status,
                            new_status,
                            change_reason
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        str(uuid.uuid4()),
                        inserted_impact_id,
                        'system',
                        None,
                        'active',
                        'Initial impact analysis creation'
                    ))

                    impacts_stored += 1
                
                if impacts_stored > 0:
                    # Update the test_cases table with project-specific impact information
                    cur.execute("""
                        WITH impact_counts AS (
                            SELECT 
                                COUNT(DISTINCT original_test_case_id) as impacted_count
                            FROM test_case_impacts
                            WHERE original_story_id = %s
                            AND project_id = %s
                            AND impact_status = 'active'
                        )
                        UPDATE test_cases 
                        SET 
                            impacted_test_cases_count = impact_counts.impacted_count,
                            last_impact_update_time = %s,
                            has_impacts = (impact_counts.impacted_count > 0)
                        FROM impact_counts
                        WHERE story_id = %s
                    """, (existing_story_id, project_id, current_time, existing_story_id))
                
                conn.commit()
                
        return impacts_stored
                
    except Exception as e:
        logger.error(f"Error storing impact analysis: {str(e)}")
        raise DatabaseError(f"Failed to store impact analysis: {str(e)}")
    finally:
        if conn:
            conn.close()

def analyze_test_case_impacts(new_story_id: str, project_id: str, existing_story_id: str = None, similarity_score: float = None, llm_ref=None):
    """
    Analyze how a new story impacts existing test cases
    Args:
        new_story_id: ID of the new story to analyze
        project_id: Project ID
        existing_story_id: Optional ID of specific story to analyze against
        similarity_score: Optional similarity score between the stories
        llm_ref: Optional LLM reference
    """
    if llm_ref is None:
        llm_ref = Config.llm_impact
        
    try:
        db_service = get_db_service()

        # Ensure required DB schema exists for impacts + history
        conn_schema = psycopg2.connect(**Config.postgres_config())
        with conn_schema:
            with conn_schema.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS test_case_impacts (
                        impact_id UUID PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        new_story_id TEXT NOT NULL,
                        original_story_id TEXT NOT NULL,
                        original_test_case_id TEXT NOT NULL,
                        modified_test_case_id TEXT NOT NULL,
                        original_run_id UUID NOT NULL,
                        impact_created_on TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        source TEXT DEFAULT 'backend',
                        similarity_score FLOAT,
                        impact_analysis_json JSONB NOT NULL,
                        previous_impact_id UUID,
                        impact_version INT DEFAULT 1,
                        impact_status TEXT DEFAULT 'active',
                        impact_type TEXT NOT NULL,
                        impact_severity TEXT NOT NULL,
                        impact_priority INTEGER,
                        impact_details JSONB
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS impact_history (
                        history_id UUID PRIMARY KEY,
                        impact_id UUID NOT NULL,
                        change_timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        changed_by TEXT,
                        previous_status TEXT,
                        new_status TEXT,
                        change_reason TEXT
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_test_case_impacts_original_story
                    ON test_case_impacts(original_story_id);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_test_case_impacts_project
                    ON test_case_impacts(project_id);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_impact_history_impact_id
                    ON impact_history(impact_id);
                """)
        conn_schema.close()
        
        # First check if stories have test cases generated
        conn = psycopg2.connect(**Config.postgres_config())
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all stories from the same project with test cases
            cur.execute("""
                SELECT story_id, test_case_generated 
                FROM test_cases 
                WHERE project_id = %s
                AND test_case_generated = TRUE
            """, (project_id,))
            generated_stories = {row['story_id']: row for row in cur.fetchall()}
        
        # Check if new story has test cases
        if new_story_id not in generated_stories:
            logger.warning(f"New story {new_story_id} does not have test cases generated yet")
            return
            
        # If analyzing specific story, check if it has test cases
        if existing_story_id:
            if existing_story_id not in generated_stories:
                logger.warning(f"Existing story {existing_story_id} does not have test cases generated yet")
                return
        
        # Get the new story's details
        new_story = db_service.get_story(new_story_id)
        if not new_story:
            raise StoryNotFoundError(f"New story {new_story_id} not found")
            
        # If existing_story_id is provided, only analyze that story
        if existing_story_id:
            existing_story = db_service.get_story(existing_story_id)
            if not existing_story:
                raise StoryNotFoundError(f"Existing story {existing_story_id} not found")
            stories_to_analyze = [existing_story]
        else:
            # Get all stories with test cases from the same project (excluding the new story)
            stories_to_analyze = [
                {"id": story_id} 
                for story_id in generated_stories.keys() 
                if story_id != new_story_id
            ]
            
        logger.info(f"Analyzing impacts for {new_story_id} against {len(stories_to_analyze)} stories from project {project_id}")
        
        def normalize_test_case_json(tc_json):
            """
            Ensure we pass a dict like:
              { "test_cases": [...] , ... }
            to the LLM prompt.
            """
            if tc_json is None:
                return None
            if isinstance(tc_json, str):
                try:
                    return json.loads(tc_json)
                except Exception:
                    logger.warning("normalize_test_case_json: failed to json.loads string")
                    return None
            if isinstance(tc_json, dict):
                return tc_json
            logger.warning(f"normalize_test_case_json: unexpected type: {type(tc_json)}")
            return None

        for existing_story in stories_to_analyze:
            try:
                # Get test cases for both stories
                new_test_cases = get_test_case_json_by_story_id(new_story_id, project_id=project_id)
                existing_test_cases = get_test_case_json_by_story_id(existing_story["id"], project_id=project_id)

                # Normalize (DB values may be stored as JSON strings or dicts)
                normalized_new = normalize_test_case_json(new_test_cases)
                normalized_existing = normalize_test_case_json(existing_test_cases)

                if not normalized_new or not normalized_existing:
                    logger.warning(
                        "Missing/invalid test_case_json for comparison; skipping",
                        extra={
                            "new_story_id": new_story_id,
                            "existing_story_id": existing_story["id"],
                            "normalized_new_type": type(normalized_new),
                            "normalized_existing_type": type(normalized_existing),
                        },
                    )
                    continue

                # LLM needs the list under test_cases
                new_list = normalized_new.get("test_cases", [])
                existing_list = normalized_existing.get("test_cases", [])
                if not isinstance(new_list, list) or not isinstance(existing_list, list):
                    logger.warning(
                        "test_case_json missing 'test_cases' list; skipping",
                        extra={
                            "new_story_id": new_story_id,
                            "existing_story_id": existing_story["id"],
                            "new_list_type": type(new_list),
                            "existing_list_type": type(existing_list),
                        },
                    )
                    continue
                    
                # Prepare the prompt for impact analysis
                # IMPORTANT: Truncate test cases to avoid exceeding LLM token limits.
                # Only send id, title, expected_result (no full steps) and cap at 10 per story.
                MAX_TCS_FOR_PROMPT = 10

                def compact_tc_list(tc_list, max_count=MAX_TCS_FOR_PROMPT):
                    """Return a compact summary list to avoid token limit errors."""
                    result = []
                    for tc in tc_list[:max_count]:
                        result.append({
                            "id": tc.get("id", tc.get("test_case_id", "")),
                            "title": tc.get("title", ""),
                            "expected_result": tc.get("expected_result", "")[:200]
                        })
                    return result

                compact_new = compact_tc_list(new_list)
                compact_existing = compact_tc_list(existing_list)

                prompt = f"""
                {IMPACT_INSTRUCTIONS}
                
                ORIGINAL STORY ({existing_story['id']} - Project: {project_id}):
                {existing_story.get('description', 'No description available')[:500]}
                
                ORIGINAL TEST CASES (summary, max {MAX_TCS_FOR_PROMPT}):
                {json.dumps(compact_existing, indent=2)}
                
                NEW STORY ({new_story_id} - Project: {project_id}):
                {new_story.get('description', 'No description available')[:500]}
                
                NEW TEST CASES (summary, max {MAX_TCS_FOR_PROMPT}):
                {json.dumps(compact_new, indent=2)}
                """
                    
                # Get impact analysis from LLM
                impact_analysis = get_llm_analysis(prompt, llm_ref)
                    
                if impact_analysis["has_impact"]:
                    # Store the impact analysis
                    impacts_stored = store_impact_analysis(
                        impact_data=impact_analysis,
                        project_id=project_id,
                        new_story_id=new_story_id,
                        existing_story_id=existing_story["id"],
                        similarity_score=similarity_score if similarity_score is not None else 0.0
                    )
                    
                    logger.info(f"Stored {impacts_stored} impacts for {existing_story['id']}")
                    
            except Exception as e:
                logger.error(f"Error analyzing impacts between {new_story_id} and {existing_story['id']}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in impact analysis for {new_story_id}: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# ... rest of the file ... 