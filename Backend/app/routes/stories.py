# Standard library imports
from datetime import datetime
import json
import uuid
import math

# Third-party imports
from flask import Blueprint, jsonify, request, send_file, Response
import psycopg2.extras
import lancedb
from dateutil import parser

# Local application imports
from app.config import Config
from app.models.db_service import get_db_service
from app.LLM.impact_analyzer import analyze_test_case_impacts
from app.utils.excel_util import generate_excel
from app.LLM.Test_case_generator import Chat_RAG
stories_bp = Blueprint('stories', __name__)

def serialize_datetime(obj):
    """Helper function to serialize datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

@stories_bp.route('/', methods=['GET'])
def get_stories():
    """Get all stories with pagination and sorting"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        sort_order = request.args.get('sort_order', 'desc')
        project_id = request.args.get('project_id')

        print("DEBUG:get_stories: Request received", {
            'page': page,
            'per_page': per_page,
            'sort_order': sort_order,
            'project_id': project_id
        })
            
        # Validate sort_order
        if sort_order not in ['asc', 'desc']:
            return jsonify({'error': 'Invalid sort_order. Must be "asc" or "desc"'}), 400

        # Get stories from PostgreSQL
        query = """
            SELECT 
                story_id as id,
                story_description as description,
                project_id,
                test_case_json,
                test_case_generated,
                created_on as test_case_created_time,
                source,
                COALESCE(impacted_test_cases_count, 0) as impacted_test_cases_count,
                COALESCE(jsonb_array_length(test_case_json->'test_cases'), 0) as test_case_count,
                inputs->>'content' as document_content,
                inputs->>'file_content' as file_content
            FROM test_cases
            WHERE test_case_generated = TRUE
        """
        params = []

        # Add project filter if specified
        if project_id:
            query += " AND project_id = %s"
            params.append(project_id)

        # Add sorting
        query += f" ORDER BY created_on {sort_order}"

        # Add pagination
        query += " LIMIT %s OFFSET %s"
        params.extend([per_page, (page - 1) * per_page])

        
        # Get embedding timestamps and doc_content_text from LanceDB
        print("DEBUG:get_stories: Connecting to LanceDB", {
            'lance_db_path': Config.LANCE_DB_PATH,
            'table_name': Config.TABLE_NAME_LANCE
        })
        db = lancedb.connect(Config.LANCE_DB_PATH)
        table = db.open_table(Config.TABLE_NAME_LANCE)
        lance_data = table.to_pandas()
        embedding_timestamps = dict(zip(lance_data['storyID'], lance_data['embedding_timestamp']))
        doc_content_texts = dict(zip(lance_data['storyID'], lance_data['doc_content_text']))
        
        # Execute query
        print("DEBUG:get_stories: Executing Postgres query", {
            'query': query,
            'params': params
        })
        with Config.get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                stories = cursor.fetchall()
                print("DEBUG:get_stories: Query returned stories", {
                    'stories_count': len(stories)
                })
                
                # Get total count for pagination
                count_query = """
                    SELECT COUNT(*) 
                    FROM test_cases 
                    WHERE test_case_generated = TRUE
                """
                count_params = []
                if project_id:
                    count_query += " AND project_id = %s"
                    count_params.append(project_id)

                cursor.execute(count_query, count_params)
                total = cursor.fetchone()[0]
                print("DEBUG:get_stories: Total stories count", {
                    'total': total
                })

                # Process results
                result = {
                    'stories': [],
                    'pagination': {
                        'total': total,
                        'page': page,
                        'per_page': per_page,
                        'total_pages': math.ceil(total / per_page)
                    }
                }

                # Format each story
                for story in stories:
                    story_dict = dict(story)
                    
                    # Add download link
                    story_dict['download_link'] = f"/api/stories/download/{story_dict['id']}"
                    
                    # Parse source info
                    source = story_dict.get('source', 'backend')
                    if isinstance(source, str):
                        try:
                            source = json.loads(source)
                        except:
                            source = {'story': source, 'test_cases': source}
                    story_dict['source'] = source

                    # Add embedding timestamp from LanceDB
                    story_dict['embedding_timestamp'] = embedding_timestamps.get(story_dict['id'])
                    if story_dict['embedding_timestamp'] and hasattr(story_dict['embedding_timestamp'], 'isoformat'):
                        story_dict['embedding_timestamp'] = story_dict['embedding_timestamp'].isoformat()

                    # Format test case creation time
                    if story_dict.get('test_case_created_time'):
                        story_dict['test_case_created_time'] = story_dict['test_case_created_time'].isoformat()

                    # Get document content from LanceDB's doc_content_text
                    story_dict['doc_content_text'] = doc_content_texts.get(story_dict['id'])

                    # Ensure impactedTestCases is included with the correct case.
                    # psycopg2 DictCursor lowercases all column names regardless of SQL aliases.
                    # The column is now aliased as 'impacted_test_cases_count' in the query.
                    def _get_impact_count(d):
                        for key in ('impactedtestcases', 'impactedTestCases', 'impacted_test_cases_count'):
                            val = d.pop(key, None) if key in d else None
                            if val is not None:
                                return int(val)
                        return 0
                    story_dict['impactedTestCases'] = _get_impact_count(story_dict)


                    result['stories'].append(story_dict)

                print("DEBUG:get_stories: Returning response", {
                    'stories_returned': len(result['stories']),
                    'pagination': result['pagination']
                })
                return jsonify(result), 200

    except Exception as e:
        print(f"❌ Error fetching stories: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/<story_id>', methods=['GET'])
def get_story(story_id):
    """Get a specific story"""
    try:
        print("DEBUG:get_story: Request received", {'story_id': story_id})
        if not story_id:
            return jsonify({'error': 'Story ID is required'}), 400

        db_service = get_db_service()
        print("DEBUG:get_story: Fetching story from db_service")
        story = db_service.get_story(story_id)
        
        if story:
            print("DEBUG:get_story: Story found", {'story_id': story_id})
            return jsonify(story)
        print("DEBUG:get_story: Story not found", {'story_id': story_id})
        return jsonify({'error': f'Story not found: {story_id}'}), 404
    except Exception as e:
        print(f"Error getting story {story_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/search', methods=['POST'])
def search_stories():
    """Search for similar stories"""
    try:
        data = request.get_json()
        print("DEBUG:search_stories: Request received", {'data': data})
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
            
        query = data['query']
        limit = data.get('limit', 5)
        
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
            
        if limit < 1:
            limit = 5
            
        db_service = get_db_service()
        print("DEBUG:search_stories: Calling db_service.search_similar_stories", {
            'query': query,
            'limit': limit
        })
        results = db_service.search_similar_stories(query, limit)
        print("DEBUG:search_stories: Returning results", {
            'results_count': len(results) if isinstance(results, list) else 'n/a'
        })
        return jsonify(results)
    except Exception as e:
        print(f"Error searching stories: {str(e)}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/testcases/download/<story_id>', methods=['GET'])
def download_test_cases(story_id):
    """Download test cases for a story as Excel file"""
    try:
        print("DEBUG:download_test_cases: Request received", {'story_id': story_id})
        db_service = get_db_service()
        
        # Get story details from LanceDB
        print("DEBUG:download_test_cases: Fetching story details from db_service")
        story = db_service.get_story(story_id)
        if not story:
            return jsonify({
                'error': f'Story not found with ID: {story_id}'
            }), 404

        # Get test cases from PostgreSQL
        print("DEBUG:download_test_cases: Querying Postgres for test_case_json")
        with psycopg2.connect(**db_service.postgres_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT test_case_json, story_description 
                    FROM test_cases 
                    WHERE story_id = %s
                """, (story_id,))
                result = cursor.fetchone()

                if not result:
                    print("DEBUG:download_test_cases: No test case row found", {'story_id': story_id})
                    return jsonify({
                        'error': f'No test cases found for story ID: {story_id}'
                    }), 404

                # Parse the test cases JSON if it's a string
                test_case_json = result[0]
                print("DEBUG:download_test_cases: Raw test_case_json type", {
                    'type': str(type(test_case_json))
                })
                if isinstance(test_case_json, str):
                    try:
                        test_case_json = json.loads(test_case_json)
                        print("DEBUG:download_test_cases: Parsed test_case_json keys", {
                            'keys': list(test_case_json.keys())
                        })
                    except json.JSONDecodeError as e:
                        print(f"Error parsing test cases JSON: {str(e)}")
                        return jsonify({
                            'error': 'Invalid test cases data format'
                        }), 500

                # Prepare data for Excel generation
                test_case_json = {
                    'storyID': story_id,
                    'storyDescription': result[1],
                    'testcases': test_case_json.get('test_cases', [])
                }
                print("DEBUG:download_test_cases: Prepared test_case_json for Excel", {
                    'storyID': story_id,
                    'testcases_count': len(test_case_json['testcases'])
                })

                # Generate Excel file
                try:
                    print("DEBUG:download_test_cases: Calling generate_excel")
                    excel_file = generate_excel(test_case_json)
                    print("DEBUG:download_test_cases: Excel generation successful, sending file")
                    return send_file(
                        excel_file,
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name=f'test_cases_{story_id}.xlsx'
                    )
                except Exception as e:
                    print(f"Error generating Excel file: {str(e)}")
                    return jsonify({
                        'error': 'Failed to generate Excel file',
                        'details': str(e)
                    }), 500

    except Exception as e:
        print(f"Error in download_test_cases: {str(e)}")
        return jsonify({
            'error': 'Failed to download test cases',
            'details': str(e)
        }), 500

@stories_bp.route('/<story_id>/testcases', methods=['GET'])
def get_story_testcases(story_id):
    """Get test cases for a specific story"""
    try:
        print(f"DEBUG:get_story_testcases: Request received for story {story_id}")
        db_service = get_db_service()
        
        # Get test cases from PostgreSQL
        print("DEBUG:get_story_testcases: Querying Postgres for test_case_json")
        with psycopg2.connect(**db_service.postgres_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT test_case_json, story_description, project_id
                    FROM test_cases
                    WHERE story_id = %s
                """, (story_id,))
                result = cur.fetchone()

                print("DEBUG:get_story_testcases: Raw DB result exists?", {
                    'has_result': bool(result),
                    'story_id': story_id
                })

                if result and result[0]:
                    print("DEBUG:get_story_testcases: Raw database result", result)
                    # Parse the test cases JSON if it's a string
                    test_case_json = result[0]
                    print("DEBUG:get_story_testcases: test_case_json type before parse", {
                        'type': str(type(test_case_json))
                    })
                    if isinstance(test_case_json, str):
                        try:
                            test_case_json = json.loads(test_case_json)
                            print("DEBUG:get_story_testcases: Parsed JSON from string", {
                                'keys': list(test_case_json.keys())
                            })
                        except json.JSONDecodeError as e:
                            print(f"Error parsing test cases JSON: {str(e)}")
                            return jsonify({
                                'error': 'Invalid test cases data format',
                                'details': str(e)
                            }), 500

                    print("DEBUG:get_story_testcases: Raw test case JSON object", test_case_json)

                    # Prepare data in the same format as the download endpoint
                    testcases_list = test_case_json.get('test_cases', [])
                    response_data = {
                        'storyID': story_id,
                        'storyDescription': result[1],
                        'project_id': result[2],
                        'testcases': testcases_list
                    }

                    print("DEBUG:get_story_testcases: Processed response data meta", {
                        'storyID': response_data['storyID'],
                        'project_id': response_data['project_id'],
                        'testcases_count': len(testcases_list)
                    })

                    return jsonify(response_data)
                
                print(f"DEBUG:get_story_testcases: No test cases found for story {story_id}")
                return jsonify({
                    'storyID': story_id,
                    'storyDescription': None,
                    'project_id': None,
                    'testcases': []
                })

    except Exception as e:
        print(f"Error getting test cases for story {story_id}: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@stories_bp.route('/next-reload', methods=['GET'])
def get_next_reload():
    try:
        # Use dynamic path instead of hardcoded path
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        next_reload_file = os.path.join(base_dir, 'next_reload.txt')
        
        if os.path.exists(next_reload_file):
            with open(next_reload_file, 'r') as f:
                content = f.read().strip()
                return jsonify({
                    'next_reload': content,
                    'source': 'file'
                }), 200
        else:
            # If file doesn't exist, return current time + 5 minutes
            from datetime import datetime, timedelta
            next_time = datetime.now() + timedelta(minutes=5)
            return jsonify({
                'next_reload': next_time.isoformat(),
                'source': 'fallback'
            }), 200
    except Exception as e:
        print(f"Error reading next reload time: {str(e)}")
        # Return explicit error with a suggested fallback time
        from datetime import datetime, timedelta
        next_time = datetime.now() + timedelta(minutes=5)
        return jsonify({
            'error': 'Failed to read next reload time',
            'details': str(e),
            'fallback_next_reload': next_time.isoformat()
        }), 500

@stories_bp.route('/trigger-reload', methods=['POST'])
def trigger_reload():
    """Trigger the scheduler to run immediately"""
    try:
        from scheduler import scheduled_job
        scheduled_job()
        return jsonify({'message': 'Scheduler triggered successfully'}), 200
    except Exception as e:
        print(f"Error triggering scheduler: {str(e)}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/rag-chat', methods=['POST'])
def rag_chat():
    """RAG chatbot endpoint: retrieves similar test cases, sends them as context to Gemini, returns generated test cases."""
    try:
        data = request.get_json()
        print("DEBUG:rag_chat: Request received", {'data': data})
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        user_query = data['query']
        # 1. Retrieve similar stories and their test cases
        try:
            print("DEBUG:rag_chat: Calling Chat_RAG", {'query': user_query})
            rag_results = Chat_RAG(user_query, top_k=3)
            print("DEBUG:rag_chat: Chat_RAG returned results", {
                'results_count': len(rag_results) if isinstance(rag_results, list) else 'n/a'
            })
        except Exception as e:
            print(f"❌ ERROR:rag_chat: Chat_RAG failed: {e}")
            return jsonify({'error': 'Failed to retrieve similar stories', 'details': str(e)}), 500
        context_cases = []
        for res in rag_results:
            tc_json = res.get('test_case_json')
            if tc_json and isinstance(tc_json, dict) and 'test_cases' in tc_json:
                context_cases.extend(tc_json['test_cases'])
        print("DEBUG:rag_chat: Context cases collected", {
            'context_cases_count': len(context_cases)
        })
        if not context_cases:
            return jsonify({'error': 'No relevant test cases found.'}), 404

        # 2. Build prompt for Gemini
        context_str = '\n'.join([
            f"Test Case {i+1}: {tc.get('title', '')}\nDescription: {tc.get('description', '')}\nSteps: {tc.get('steps', [])}\nExpected Result: {tc.get('expected_result', '')}\n" for i, tc in enumerate(context_cases)
        ])
        prompt = f"""
You are an experienced QA analyst. Here are test cases from similar stories:
{context_str}

Now, based on the following user story, generate new, comprehensive test cases in JSON format (fields: id, title, steps, expected_result, priority):
{user_query}
"""

        # 3. Call Gemini LLM using the configured object
        try:
            print("DEBUG:rag_chat: Calling LLM via Config.llm.invoke")
            response = Config.llm.invoke(prompt)
            text = response.content.strip()
            print("DEBUG:rag_chat: LLM raw output received", {
                'length': len(text)
            })
        except Exception as e:
            print(f"❌ ERROR:rag_chat: LLM invocation failed: {e}")
            return jsonify({'error': 'Failed to generate test cases from LLM', 'details': str(e)}), 500
        # Clean triple backticks and ```json
        cleaned = text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        import json
        try:
            print("DEBUG:rag_chat: Attempting to parse LLM output as JSON")
            parsed = json.loads(cleaned)
            print("DEBUG:rag_chat: Parsed LLM JSON type", {'type': str(type(parsed))})
            # If it's a list of test cases, return as JSON
            if isinstance(parsed, list):
                print("DEBUG:rag_chat: Returning list of test cases", {
                    'count': len(parsed)
                })
                return jsonify({'testCases': parsed})
            # If it's a dict with 'test_cases', return that
            if isinstance(parsed, dict) and 'test_cases' in parsed:
                print("DEBUG:rag_chat: Returning dict['test_cases']", {
                    'count': len(parsed['test_cases'])
                })
                return jsonify({'testCases': parsed['test_cases']})
            # Otherwise, return the parsed object
            print("DEBUG:rag_chat: Returning parsed object without transformation")
            return jsonify({'testCases': parsed})
        except Exception as e:
            print("Failed to parse LLM output as JSON:", e)
            return jsonify({'raw': cleaned, 'error': 'Failed to parse LLM output as JSON', 'details': str(e)}), 200
    except Exception as e:
        print(f"Error in rag_chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/projects', methods=['GET'])
def get_projects():
    """Get unique project IDs"""
    try:
        db_service = get_db_service()
        
        # Get stories from LanceDB
        stories_table = db_service.lance_db.open_table(Config.TABLE_NAME_LANCE)
        lance_stories = stories_table.to_pandas()
        
        # Get unique project IDs
        project_ids = lance_stories['project_id'].dropna().unique().tolist()
        project_ids = [pid for pid in project_ids if pid and pid.strip()]
        
        return jsonify({
            'projects': project_ids
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/upload', methods=['POST'])
def upload_story():
    """Upload a new user story with project ID, story ID, content, and optional file. Description will be AI-generated."""
    try:
        # Get form data
        project_id = request.form.get('project_id')
        story_id = request.form.get('story_id')
        content = request.form.get('content')  # Changed from description to content
        file = request.files.get('file')
        source = request.form.get('source', 'backend')  # Default to 'backend' if not provided

        # Validate required fields
        if not project_id or not story_id or (not content and not file):
            return jsonify({
                'error': 'Project ID, Story ID, and either Content or File are required'
            }), 400

        # Check if story already exists
        db_service = get_db_service()
        existing_story = db_service.get_story(story_id)
        if existing_story:
            return jsonify({
                'error': f'Story with ID "{story_id}" already exists'
            }), 409

        # Create project folder if it doesn't exist
        import os
        upload_dir = os.getenv("UPLOAD_FOLDER", "./data/uploaded_docs")
        project_dir = os.path.join(upload_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)

        # Handle file upload if provided
        file_path = None
        file_content = None
        
        if file and file.filename:
            # Validate file type
            allowed_extensions = {'.pdf', '.docx', '.txt'}
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                return jsonify({
                    'error': f'File type {file_ext} not supported. Allowed: {", ".join(allowed_extensions)}'
                }), 400

            # Save file to project folder
            filename = f"{story_id}{file_ext}"
            file_path = os.path.join(project_dir, filename)
            file.save(file_path)
            
            # Extract text from file
            try:
                from app.datapipeline.text_extractor import extract_text
                file_content = extract_text(file_path)
                if not file_content:
                    return jsonify({
                        'error': 'Could not extract text from the uploaded file'
                    }), 400
            except Exception as e:
                return jsonify({
                    'error': f'Error extracting text from file: {str(e)}'
                }), 500

        # Create story content (use file content if available, otherwise use typed content)
        story_content = file_content if file_content else content

        # Generate AI description from content using chunking method
        try:
            def summarize_in_chunks(text, chunk_size=4000):
                try:
                    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                    summaries = []
                    for chunk in chunks[:3]:  # Limit to 3 chunks for efficiency
                        prompt = (
                            "Summarize the following document section in 1 sentence:\n\n" + chunk
                        )
                        try:
                            response = Config.llm.invoke(prompt)
                            summaries.append(response.content.strip())
                        except Exception as e:
                            summaries.append("[Summary failed for a chunk]")
                            print(f"❌ LLM failed on a chunk: {e}")
                    return " ".join(summaries)
                except Exception as e:
                    print(f"❌ LLM summary failed: {e}")
                    return "Summary could not be generated."

            # Generate description using the chunking method
            description = summarize_in_chunks(story_content)
                
        except Exception as e:
            print(f"Error generating AI description: {str(e)}")
            # Fallback to a simple truncation of content
            description = story_content[:147] + "..." if len(story_content) > 150 else story_content

        # Add to LanceDB
        try:
            from app.config import EMBEDDING_MODEL
            from datetime import datetime
            import lancedb
            
            db = lancedb.connect(Config.LANCE_DB_PATH)
            table = db.open_table(Config.TABLE_NAME_LANCE)
            
            # Generate embedding
            embedding = EMBEDDING_MODEL.encode(story_content).tolist()
            
            # Add to LanceDB
            table.add([{
                "project_id": project_id,
                "vector": embedding,
                "storyID": story_id,
                "storyDescription": description,  # AI-generated description
                "test_case_content": "",
                "filename": filename if file else f"{story_id}.txt",
                "original_path": file_path if file else None,
                "doc_content_text": story_content,
                "embedding_timestamp": datetime.now(),
                "source": source  # Add source field
            }])
            
        except Exception as e:
            return jsonify({
                'error': f'Error adding story to database: {str(e)}'
            }), 500

        # Generate test cases
        try:
            from app.LLM.Test_case_generator import generate_test_case_for_story
            generate_test_case_for_story(story_id)
            
            # Check if test cases were generated successfully
            from app.models.postgress_writer import get_test_case_json_by_story_id
            test_cases_result = get_test_case_json_by_story_id(story_id)
            
            if test_cases_result:
                # Move file to success folder
                if file_path and os.path.exists(file_path):
                    success_dir = os.path.join(os.getenv("SUCCESS_FOLDER", "./data/success"), project_id)
                    os.makedirs(success_dir, exist_ok=True)
                    import shutil
                    shutil.move(file_path, os.path.join(success_dir, filename))
                
                # -------------------------------------------------------
                # Run impact analysis in background thread:
                # This finds all existing stories in the same project and
                # updates their impacted_test_cases_count so the table
                # immediately reflects how many test cases are affected.
                # -------------------------------------------------------
                import threading
                def run_impact_analysis(new_story_id, proj_id):
                    try:
                        print(f"[ImpactAnalysis] Starting background impact analysis for story {new_story_id} in project {proj_id}")
                        from app.LLM.impact_analyzer import analyze_test_case_impacts
                        analyze_test_case_impacts(
                            new_story_id=new_story_id,
                            project_id=proj_id
                        )
                        print(f"[ImpactAnalysis] ✅ Background impact analysis completed for story {new_story_id}")
                    except Exception as ia_err:
                        print(f"[ImpactAnalysis] ❌ Background impact analysis failed for story {new_story_id}: {ia_err}")
                
                impact_thread = threading.Thread(
                    target=run_impact_analysis,
                    args=(story_id, project_id),
                    daemon=True
                )
                impact_thread.start()
                print(f"[Upload] Impact analysis thread started for story {story_id}")
                
                return jsonify({
                    'message': 'Story uploaded and test cases generated successfully',
                    'story_id': story_id,
                    'project_id': project_id,
                    'description': description,  # Return the AI-generated description
                    'test_cases_generated': True,
                    'file_processed': file is not None,
                    'source': source
                }), 200
            else:
                # Move file to failure folder if test case generation failed
                if file_path and os.path.exists(file_path):
                    failure_dir = os.path.join(os.getenv("FAILURE_FOLDER", "./data/failure"), project_id)
                    os.makedirs(failure_dir, exist_ok=True)
                    import shutil
                    shutil.move(file_path, os.path.join(failure_dir, filename))
                
                return jsonify({
                    'message': 'Story uploaded but test case generation failed',
                    'story_id': story_id,
                    'project_id': project_id,
                    'description': description,  # Return the AI-generated description
                    'test_cases_generated': False,
                    'file_processed': file is not None,
                    'source': source
                }), 200
                
        except Exception as e:
            # Move file to failure folder
            if file_path and os.path.exists(file_path):
                failure_dir = os.path.join(os.getenv("FAILURE_FOLDER", "./data/failure"), project_id)
                os.makedirs(failure_dir, exist_ok=True)
                import shutil
                shutil.move(file_path, os.path.join(failure_dir, filename))
            
            return jsonify({
                'error': f'Error generating test cases: {str(e)}'
            }), 500

    except Exception as e:
        print(f"Error in upload_story: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@stories_bp.route('/impacts/<project_id>', methods=['GET'])
def get_project_impacts(project_id):
    """Get all impact analyses for a project"""
    try:
        db_service = get_db_service()
        # Use Config's connection method with context managers
        with Config.get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        tci.impact_id,
                        tci.new_story_id,
                        tci.original_story_id,
                        tci.original_test_case_id,
                        tci.modified_test_case_id,
                        tci.impact_created_on,
                        tci.similarity_score,
                        tci.impact_analysis_json,
                        tc.story_description as original_story_description
                    FROM test_case_impacts tci
                    JOIN test_cases tc ON tc.story_id = tci.original_story_id
                    WHERE tci.project_id = %s
                    ORDER BY tci.impact_created_on DESC
                """, (project_id,))
                
                impacts = cursor.fetchall()
                
                return jsonify({
                    'project_id': project_id,
                    'total_impacts': len(impacts),
                    'impacts': [dict(impact) for impact in impacts]
                }), 200
                
    except Exception as e:
        print(f"❌ Error fetching project impacts: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/impacts/story/<story_id>', methods=['GET'])
def get_story_impacts(story_id):
    """Get impacts where this story is either the source or target"""
    try:
        db_service = get_db_service()
        # Use Config's connection method with context managers
        with Config.get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        tci.*,
                        tc_orig.story_description as impacted_story_description,
                        tc_new.story_description as new_story_description
                    FROM test_case_impacts tci
                    JOIN test_cases tc_orig ON tc_orig.story_id = tci.original_story_id
                    JOIN test_cases tc_new ON tc_new.story_id = tci.new_story_id
                    WHERE tci.original_story_id = %s OR tci.new_story_id = %s
                    ORDER BY tci.impact_created_on DESC
                """, (story_id, story_id))
                
                impacts = cursor.fetchall()
                
                # Organize impacts by role (source vs target)
                caused_impacts = []
                received_impacts = []
                
                for impact in impacts:
                    impact_dict = dict(impact)
                    if impact['new_story_id'] == story_id:
                        caused_impacts.append(impact_dict)
                    else:
                        received_impacts.append(impact_dict)
                
                return jsonify({
                    'story_id': story_id,
                    'total_impacts': len(impacts),
                    'caused_impacts': caused_impacts,
                    'received_impacts': received_impacts
                }), 200
                
    except Exception as e:
        print(f"❌ Error fetching story impacts: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/impacts/details/<impact_id>', methods=['GET'])
def get_impact_details(impact_id):
    """Get detailed view of a specific impact"""
    try:
        db_service = get_db_service()
        # Use Config's connection method with context managers
        with Config.get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        tci.*,
                        tc_orig.story_description as original_story_description,
                        tc_orig.test_case_json as original_test_cases,
                        tc_new.story_description as new_story_description,
                        tc_new.test_case_json as new_test_cases
                    FROM test_case_impacts tci
                    JOIN test_cases tc_orig ON tc_orig.story_id = tci.original_story_id
                    JOIN test_cases tc_new ON tc_new.story_id = tci.new_story_id
                    WHERE tci.impact_id = %s
                """, (impact_id,))
                
                impact = cursor.fetchone()
                
                if not impact:
                    return jsonify({'error': 'Impact not found'}), 404
                    
                return jsonify(dict(impact)), 200
                
    except Exception as e:
        print(f"❌ Error fetching impact details: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/impacts/summary/<project_id>', methods=['GET'])
def get_project_impact_summary(project_id):
    """Get summary of impacts in a project"""
    try:
        db_service = get_db_service()
        cursor = db_service.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get summary from our view
        cursor.execute("""
            SELECT * FROM test_case_impact_summary
            WHERE project_id = %s
            ORDER BY total_impacts DESC
        """, (project_id,))
        
        summaries = cursor.fetchall()
        cursor.close()
        
        return jsonify({
            'project_id': project_id,
            'total_stories': len(summaries),
            'stories_with_impacts': len([s for s in summaries if s['total_impacts'] > 0]),
            'summaries': [dict(summary) for summary in summaries]
        }), 200
        
    except Exception as e:
        print(f"❌ Error fetching impact summary: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/impacts/analyze', methods=['POST'])
async def trigger_impact_analysis():
    """Manually trigger impact analysis for a story"""
    try:
        data = request.get_json()
        story_id = data.get('story_id')
        project_id = data.get('project_id')
        
        if not story_id or not project_id:
            return jsonify({
                'error': 'Both story_id and project_id are required'
            }), 400
            
        # Run impact analysis
        await analyze_test_case_impacts(story_id, project_id)
        
        return jsonify({
            'message': 'Impact analysis triggered successfully',
            'story_id': story_id,
            'project_id': project_id
        }), 200
        
    except Exception as e:
        print(f"❌ Error triggering impact analysis: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/impacts/analyze-project', methods=['POST'])
def trigger_project_impact_analysis():
    """Trigger impact analysis for ALL stories in a project (background thread).
    Refreshes impacted_test_cases_count for each story so the table shows
    accurate non-zero counts after a new story is uploaded.
    """
    try:
        data = request.get_json()
        project_id = data.get('project_id')
        
        if not project_id:
            return jsonify({'error': 'project_id is required'}), 400
        
        # Get all stories with test cases in this project
        with Config.get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT story_id FROM test_cases
                    WHERE project_id = %s AND test_case_generated = TRUE
                    ORDER BY created_on ASC
                """, (project_id,))
                stories = [row['story_id'] for row in cursor.fetchall()]
        
        if not stories:
            return jsonify({
                'message': 'No stories with test cases found in project',
                'project_id': project_id,
                'stories_queued': 0
            }), 200
        
        # Run analysis in background thread
        import threading
        def run_all_impact_analyses(story_list, proj_id):
            try:
                print(f"[ImpactAnalysis] Starting project-wide impact analysis for {len(story_list)} stories in project {proj_id}")
                from app.LLM.impact_analyzer import analyze_test_case_impacts
                for sid in story_list:
                    try:
                        analyze_test_case_impacts(new_story_id=sid, project_id=proj_id)
                        print(f"[ImpactAnalysis] Completed impact analysis for story {sid}")
                    except Exception as story_err:
                        print(f"[ImpactAnalysis] Error analyzing story {sid}: {story_err}")
                print(f"[ImpactAnalysis] Project-wide impact analysis completed for project {proj_id}")
            except Exception as e:
                print(f"[ImpactAnalysis] Project-wide impact analysis failed: {e}")
        
        thread = threading.Thread(
            target=run_all_impact_analyses,
            args=(stories, project_id),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'message': f'Impact analysis started for {len(stories)} stories in background',
            'project_id': project_id,
            'stories_queued': len(stories)
        }), 202
        
    except Exception as e:
        print(f"Error triggering project impact analysis: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/<story_id>/test-cases/<test_case_id>', methods=['GET'])
def get_test_case(story_id, test_case_id):
    """Get a specific test case from a story"""
    try:
        print(f"Fetching test case {test_case_id} from story {story_id}")
        db_service = get_db_service()
        
        # Get test cases from PostgreSQL
        with psycopg2.connect(**db_service.postgres_config) as conn:
            with conn.cursor() as cur:
                # Use a JSON path query to extract the specific test case
                cur.execute("""
                    WITH test_cases_array AS (
                        SELECT jsonb_array_elements(test_case_json->'test_cases') as test_case
                        FROM test_cases
                        WHERE story_id = %s
                    )
                    SELECT test_case
                    FROM test_cases_array
                    WHERE test_case->>'id' = %s
                    OR test_case->>'test_case_id' = %s
                    LIMIT 1
                """, (story_id, test_case_id, test_case_id))
                result = cur.fetchone()

                if not result:
                    return jsonify({
                        'error': f'Test case {test_case_id} not found in story {story_id}'
                    }), 404

                test_case = result[0]
                return jsonify({
                    'id': test_case.get('id') or test_case.get('test_case_id'),
                    'title': test_case.get('title'),
                    'test_steps': test_case.get('steps', []),
                    'expected_result': test_case.get('expected_result', ''),
                    'priority': test_case.get('priority', 'medium'),
                    'severity': test_case.get('severity', 'medium')
                })

    except Exception as e:
        print(f"Error getting test case {test_case_id} from story {story_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/impacts/story-test-cases/<story_id>', methods=['GET'])
def get_story_test_case_impacts(story_id):
    """Get details of how test cases in this story were impacted by other stories"""
    try:
        project_id = request.args.get('project_id')
        if not project_id:
            return jsonify({'error': 'project_id is required as query parameter'}), 400

        print(f"DEBUG: Starting story test case impacts fetch for story {story_id} in project {project_id}")
        
        # Get story description first to check if story exists
        with Config.get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # First get the impacted test cases count directly from the database
                cursor.execute("""
                    SELECT COUNT(DISTINCT original_test_case_id) as impacted_count
                    FROM test_case_impacts
                    WHERE original_story_id = %s
                    AND project_id = %s
                    AND impact_status = 'active'
                """, (story_id, project_id))
                db_impacted_count = cursor.fetchone()['impacted_count']

                cursor.execute("""
                    SELECT story_description, test_case_json
                    FROM test_cases
                    WHERE story_id = %s
                """, (story_id,))
                story_info = cursor.fetchone()

                if not story_info:
                    return jsonify({'error': 'Story not found'}), 404

                print("DEBUG: Found story info")
                story_info = dict(story_info)
                
                # Handle test_case_json which could be either string or dict
                test_case_json = story_info['test_case_json']
                if isinstance(test_case_json, str):
                    try:
                        test_case_json = json.loads(test_case_json)
                    except Exception as e:
                        print(f"DEBUG: Error parsing test_case_json string: {e}")
                        test_case_json = {'test_cases': []}
                elif not isinstance(test_case_json, dict):
                    print(f"DEBUG: Unexpected test_case_json type: {type(test_case_json)}")
                    test_case_json = {'test_cases': []}
                
                test_cases = test_case_json.get('test_cases', [])
                total_test_cases = len(test_cases)
                test_case_ids = [tc.get('id') for tc in test_cases]
                print(f"DEBUG: Found {total_test_cases} test cases with IDs: {test_case_ids}")

                # Get all impacts for these test cases
                if test_case_ids:
                    placeholders = ','.join(['%s'] * len(test_case_ids))
                    cursor.execute(f"""
                        SELECT 
                            tci.original_test_case_id,
                            tci.impact_id,
                            tci.new_story_id,
                            new_tc.story_description,
                            tci.impact_created_on,
                            tci.impact_severity,
                            tci.impact_priority,
                            tci.impact_details,
                            tci.impact_analysis_json
                        FROM test_case_impacts tci
                        JOIN test_cases new_tc ON new_tc.story_id = tci.new_story_id
                        WHERE tci.project_id = %s 
                        AND tci.original_test_case_id IN ({placeholders})
                        AND tci.impact_status = 'active'
                        ORDER BY tci.original_test_case_id, tci.impact_created_on DESC
                    """, (project_id,) + tuple(test_case_ids))
                    impacts = cursor.fetchall()
                    print(f"DEBUG: Found {len(impacts)} impacts")
                else:
                    impacts = []

                # Group impacts by test case ID
                impact_map = {}
                for impact in impacts:
                    test_case_id = impact['original_test_case_id']
                    if test_case_id not in impact_map:
                        impact_map[test_case_id] = []
                    
                    # Handle impact_analysis_json
                    impact_analysis = impact['impact_analysis_json']
                    try:
                        if isinstance(impact_analysis, str):
                            impact_analysis = json.loads(impact_analysis)
                        elif hasattr(impact_analysis, 'decode'):
                            impact_analysis = json.loads(impact_analysis.decode())
                    except Exception as e:
                        print(f"DEBUG: Error parsing impact_analysis_json: {e}")
                        impact_analysis = str(impact_analysis)

                    formatted_impact = {
                        'impact_id': str(impact['impact_id']),
                        'impacting_story_id': str(impact['new_story_id']),
                        'impacting_story_description': str(impact['story_description']) if impact['story_description'] else '',
                        'impact_created_on': impact['impact_created_on'].isoformat() if isinstance(impact['impact_created_on'], datetime) else str(impact['impact_created_on']) if impact['impact_created_on'] else None,
                        'impact_severity': str(impact['impact_severity']) if impact['impact_severity'] else '',
                        'impact_priority': str(impact['impact_priority']) if impact['impact_priority'] else '',
                        'impact_details': str(impact['impact_details']) if impact['impact_details'] else '',
                        'impact_analysis_json': json.dumps(impact_analysis) if isinstance(impact_analysis, (dict, list)) else str(impact_analysis)
                    }
                    impact_map[test_case_id].append(formatted_impact)

                # Build response with all test cases
                impacted_test_cases = []
                for tc in test_cases:
                    tc_id = tc.get('id')
                    steps = tc.get('steps', [])
                    steps_str = json.dumps(steps) if isinstance(steps, (list, dict)) else str(steps)
                    
                    test_case = {
                        'test_case_id': str(tc_id),
                        'original_title': str(tc.get('title', '')),
                        'original_expected_result': str(tc.get('expected_result', '')),
                        'original_steps': steps_str,
                        'impacts': impact_map.get(tc_id, [])  # Get impacts from map if they exist, empty list if not
                    }
                    impacted_test_cases.append(test_case)

                # Use the database count instead of calculating it here
                print(f"DEBUG: Using database count - {db_impacted_count} impacted test cases")
                
                # Create response structure
                response = {
                    'story_id': str(story_id),
                    'story_description': str(story_info['story_description']) if story_info['story_description'] else '',
                    'total_test_cases': str(total_test_cases),
                    'impacted_test_cases_count': str(db_impacted_count),
                    'impact_percentage': str(round((db_impacted_count / total_test_cases * 100) if total_test_cases > 0 else 0, 2)),
                    'impacted_test_cases': impacted_test_cases,
                    'severity_summary': {
                        'high': str(len([tc for tc in impacted_test_cases if any(i['impact_severity'] == 'high' for i in tc['impacts'])])),
                        'medium': str(len([tc for tc in impacted_test_cases if any(i['impact_severity'] == 'medium' for i in tc['impacts'])])),
                        'low': str(len([tc for tc in impacted_test_cases if any(i['impact_severity'] == 'low' for i in tc['impacts'])]))
                    }
                }
                
                return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error fetching story test case impacts: {e}")
        return jsonify({'error': str(e)}), 500

@stories_bp.route('/impacts/details/<story_id>', methods=['GET'])
def get_story_impact_details(story_id):
    """Get detailed impact information for a story"""
    try:
        project_id = request.args.get('project_id')
        if not project_id:
            return jsonify({'error': 'project_id is required as query parameter'}), 400

        db_service = get_db_service()
        with Config.get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Get story details first
                cursor.execute("""
                    SELECT story_description, test_case_json
                    FROM test_cases
                    WHERE story_id = %s
                """, (story_id,))
                story_info = cursor.fetchone()

                if not story_info:
                    return jsonify({'error': 'Story not found'}), 404

                # Get all test cases and their impacts
                cursor.execute("""
                    WITH story_test_cases AS (
                        SELECT 
                            story_id,
                            jsonb_array_elements(test_case_json->'test_cases') as test_case,
                            (jsonb_array_elements(test_case_json->'test_cases')->>'id') as test_case_id
                        FROM test_cases
                        WHERE story_id = %s
                    ),
                    impact_details AS (
                        SELECT 
                            tc.test_case_id,
                            tc.test_case->>'title' as test_case_title,
                            tc.test_case->>'expected_result' as original_expected_result,
                            tc.test_case->'steps' as original_steps,
                            json_agg(
                                CASE WHEN tci.impact_id IS NOT NULL THEN
                                    json_build_object(
                                        'impact_id', tci.impact_id,
                                        'impacting_story_id', tci.new_story_id,
                                        'impact_created_on', tci.impact_created_on,
                                        'impact_severity', tci.impact_severity,
                                        'impact_priority', tci.impact_priority,
                                        'impact_details', tci.impact_details,
                                        'impact_analysis_json', tci.impact_analysis_json
                                    )
                                ELSE NULL
                                END
                            ) FILTER (WHERE tci.impact_id IS NOT NULL) as impacts
                        FROM story_test_cases tc
                        LEFT JOIN test_case_impacts tci ON tci.original_test_case_id = tc.test_case_id
                            AND tci.project_id = %s
                            AND tci.impact_status = 'active'
                        GROUP BY tc.test_case_id, tc.test_case
                    )
                    SELECT 
                        test_case_id,
                        test_case_title,
                        original_expected_result,
                        original_steps,
                        impacts,
                        impacts IS NOT NULL as is_impacted,
                        CASE 
                            WHEN impacts IS NULL THEN 'none'
                            WHEN impacts::text LIKE '%%"impact_severity":"high"%%' THEN 'high'
                            WHEN impacts::text LIKE '%%"impact_severity":"medium"%%' THEN 'medium'
                            ELSE 'low'
                        END as highest_severity
                    FROM impact_details
                    ORDER BY 
                        is_impacted DESC,
                        CASE highest_severity 
                            WHEN 'high' THEN 1
                            WHEN 'medium' THEN 2
                            WHEN 'low' THEN 3
                            ELSE 4
                        END,
                        test_case_title
                """, (story_id, project_id))
                
                test_cases = cursor.fetchall()

                # Calculate summary statistics
                raw_tc_json = story_info['test_case_json']
                try:
                    if isinstance(raw_tc_json, str):
                        parsed_tc_json = json.loads(raw_tc_json)
                    elif isinstance(raw_tc_json, dict):
                        parsed_tc_json = raw_tc_json
                    else:
                        print(f"DEBUG:get_story_impact_details: Unexpected test_case_json type: {type(raw_tc_json)}")
                        parsed_tc_json = {'test_cases': []}
                except Exception as e:
                    print(f"DEBUG:get_story_impact_details: Error parsing test_case_json: {e}")
                    return jsonify({
                        'error': 'Invalid test cases data format for story',
                        'details': str(e)
                    }), 500

                total_test_cases = len(parsed_tc_json.get('test_cases', []))
                impacted_test_cases = sum(1 for tc in test_cases if tc['is_impacted'])
                severity_counts = {
                    'high': sum(1 for tc in test_cases if tc['highest_severity'] == 'high'),
                    'medium': sum(1 for tc in test_cases if tc['highest_severity'] == 'medium'),
                    'low': sum(1 for tc in test_cases if tc['highest_severity'] == 'low')
                }

                response = {
                    'story_id': story_id,
                    'story_description': story_info['story_description'],
                    'total_test_cases': total_test_cases,
                    'impacted_test_cases_count': impacted_test_cases,
                    'impact_percentage': round((impacted_test_cases / total_test_cases * 100) if total_test_cases > 0 else 0, 2),
                    'severity_summary': severity_counts,
                    'test_cases': [
                        {
                            'test_case_id': tc['test_case_id'],
                            'title': tc['test_case_title'],
                            'expected_result': tc['original_expected_result'],
                            'steps': tc['original_steps'],
                            'is_impacted': tc['is_impacted'],
                            'highest_severity': tc['highest_severity'],
                            'impacts': tc['impacts'] if tc['impacts'] else []
                        } for tc in test_cases
                    ]
                }

                return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error fetching story impact details: {e}")
        return jsonify({'error': str(e)}), 500 