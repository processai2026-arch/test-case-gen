import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os
import uuid
import datetime
import json
from app.config import Config

load_dotenv()


def get_test_case_json_by_story_id(story_id, project_id=None):
    """Get latest generated test_case_json for a story_id (optionally project-scoped)."""
    conn = None
    try:
        conn = Config.get_postgres_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            query = """
                SELECT test_case_json
                FROM test_cases
                WHERE story_id = %s
                AND test_case_json IS NOT NULL
                AND test_case_generated = TRUE
            """
            params = [story_id]
            if project_id:
                query += " AND project_id = %s"
                params.append(project_id)
            query += """
                ORDER BY created_on DESC NULLS LAST
                LIMIT 1
            """
            cur.execute(query, tuple(params))
            result = cur.fetchone()
            return result["test_case_json"] if result else None
    except Exception as e:
        print(f"❌ Error fetching test case for {story_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_generated_story_ids():
    """Fetch list of story_ids that already have test cases generated."""
    try:
        conn = Config.get_postgres_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT story_id FROM test_cases
                WHERE test_case_generated = TRUE
            """)
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"❌ Error fetching generated story IDs: {e}")
        return []
    finally:
        if conn:
            conn.close()

def insert_test_case(story_id, story_description, test_case_json, project_id=None, source='backend', inputs=None):
    """Insert or update generated test case JSON into PostgreSQL."""
    try:
        conn = Config.get_postgres_connection()
        with conn.cursor() as cur:
            # First check if a record exists for this story_id
            cur.execute("SELECT run_id FROM test_cases WHERE story_id = %s", (story_id,))
            existing = cur.fetchone()
            
            run_id = existing[0] if existing else str(uuid.uuid4())
            created_on = datetime.datetime.now()
            total_test_cases = len(test_case_json.get("test_cases", []))

            query = """
                INSERT INTO test_cases (
                    project_id,
                    run_id,
                    story_id,
                    story_description,
                    created_on,
                    test_case_json,
                    total_test_cases,
                    test_case_generated,
                    source,
                    inputs
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    project_id = EXCLUDED.project_id,
                    story_description = EXCLUDED.story_description,
                    created_on = EXCLUDED.created_on,
                    test_case_json = EXCLUDED.test_case_json,
                    total_test_cases = EXCLUDED.total_test_cases,
                    test_case_generated = TRUE,
                    source = EXCLUDED.source,
                    inputs = EXCLUDED.inputs
            """

            cur.execute(query, (
                project_id,
                run_id,
                story_id,
                story_description,
                created_on,
                json.dumps(test_case_json),
                total_test_cases,
                source,
                json.dumps(inputs) if inputs else None
            ))
            conn.commit()
    except Exception as e:
        print(f"❌ Failed to insert test case for {story_id}: {e}")
    finally:
        if conn:
            conn.close()