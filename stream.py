import streamlit as st
import os
import fitz  # PyMuPDF
import json
from dotenv import load_dotenv

# For Gemini integration
import google.generativeai as genai
import google.api_core.exceptions # For more specific Gemini error types

# Load .env file if present (for local development convenience)
load_dotenv()

# --- AIQuestionGenerator Class (Gemini-Only) ---
class AIQuestionGenerator:
    def __init__(self):
        self.model_type = "gemini"
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.model_name = None
        self.is_configured = False

        # st.write(f"Attempting to initialize AIQuestionGenerator with type: {self.model_type}") # Less verbose
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
                self.client = genai.GenerativeModel(self.model_name)
                # Test call to verify client (optional, but good for immediate feedback)
                # self.client.generate_content("test", generation_config=genai.types.GenerationConfig(max_output_tokens=5))
                st.sidebar.success(f"Gemini initialized with {self.model_name}")
                self.is_configured = True
            except Exception as e:
                st.sidebar.error(f"Gemini Initialization Error: {e}")
        else:
            st.sidebar.warning("GEMINI_API_KEY not found. Set it in the sidebar.")

    def _return_error_structure(self, error_message_for_ui, error_message_for_dict=None):
        """Helper to return consistent error structure."""
        st.error(error_message_for_ui) # Display error in Streamlit UI
        final_error_msg = error_message_for_dict if error_message_for_dict else error_message_for_ui
        return {
            "technical": [f"Error: {final_error_msg}"],
            "behavioral": [],
            "project_specific": [],
            "scenario_based": []
        }

    def generate_questions_from_text(self, resume_text_content: str, job_description_text: str | None = None):
        if not self.client or not self.is_configured:
            return self._return_error_structure(
                "Gemini client not configured. Please check API key.",
                "Gemini client not configured."
            )

        job_description_section = "No job description provided."
        if job_description_text and job_description_text.strip():
            job_description_section = f"""
        **Job Description:**
        ---
        {job_description_text}
        ---"""

        prompt = f"""
        You are an expert technical interviewer and career coach. Your task is to generate insightful interview questions based on the provided resume and job description. Aim for a comprehensive set of at least 25 questions, tailored to help a candidate prepare for internship interviews.

        **Input Materials:**
        ---
        **Resume Content:**
        {resume_text_content}
        ---
        {job_description_section}
        ---

        **Question Generation Guidelines:**
        1.  **Relevance:** Prioritize questions that directly relate to skills, technologies, and experiences mentioned in BOTH the resume and the job description. If no JD, focus on resume.
        2.  **Depth & Breadth:** Cover a range of topics from fundamental concepts to practical application.
        3.  **Realism:** Formulate questions similar to those asked in actual internship interviews for tech roles.
        4.  **Progression:** Include questions suitable for different interview stages (screening, technical deep-dive).
        5.  **Clarity:** Ensure questions are unambiguous.

        **Question Categories (ensure a good distribution, aiming for 25+ total):**

        1.  **Technical Deep Dive (8-10 questions):**
            *   Based on specific programming languages, frameworks, and tools listed (e.g., Python, React, Docker, Git).
            *   Data structures and algorithms (e.g., "Explain how you would use a hash map to optimize [specific scenario from resume/JD]?").
            *   Database concepts (SQL/NoSQL, querying, design if mentioned).
            *   System design fundamentals (scaled appropriately for an intern, e.g., "How would you design a simple URL shortener, focusing on the API and data storage?").
            *   Debugging and troubleshooting approaches.

        2.  **Project-Specific (6-8 questions):**
            *   Probe into specific projects listed on the resume.
            *   "On project X, you mentioned using Y technology. Can you walk me through a specific challenge you faced and how you overcame it?"
            *   "What was your specific contribution to project Z? How did you collaborate with others?"
            *   "If you could redo project A, what would you do differently and why?"
            *   "How did you test your work on project B?"

        3.  **Behavioral & Situational (6-8 questions):** (Use STAR method for answering these)
            *   "Describe a time you had to learn a new technology quickly for a project."
            *   "Tell me about a challenging team project and how you handled disagreements."
            *   "How do you approach a task when the requirements are unclear?"
            *   "Describe a mistake you made and what you learned from it."
            *   (If JD is present) "This role requires [skill from JD, e.g., 'strong problem-solving skills']. Can you give an example of how you've demonstrated this?"

        4.  **Scenario-Based/Problem-Solving (4-6 questions):**
            *   "Imagine you're given a task to build [a small feature related to JD or resume skills]. What would be your initial steps and thought process?"
            *   "How would you debug a situation where [common problem, e.g., 'a web application is running slower than expected']?"
            *   "You've pushed code that inadvertently broke a feature in production for a personal project. What steps would you take immediately?"

        **Output Format:**
        Return a single, valid JSON object with the following structure. Do NOT include any text before or after the JSON object (e.g. no "```json" or "```").

        {{
            "technical": ["Question 1 about tech...", "Question 2 about tech..."],
            "behavioral": ["Question 1 about behavior...", "Question 2 about behavior..."],
            "project_specific": ["Question 1 about project...", "Question 2 about project..."],
            "scenario_based": ["Scenario question 1...", "Scenario question 2..."]
        }}
        """

        try:
            response = self.client.generate_content(prompt)

            # Detailed checks for Gemini response issues
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = str(response.prompt_feedback.block_reason)
                ratings = "\n".join([f"  - {r.category}: {r.probability}" for r in response.prompt_feedback.safety_ratings])
                return self._return_error_structure(
                    f"Gemini API blocked the prompt. Reason: {reason}.\nSafety Ratings:\n{ratings}",
                    f"Prompt blocked by safety settings (Reason: {reason})."
                )

            if not response.candidates:
                return self._return_error_structure(
                    "Gemini API returned no candidates in the response.",
                    "No candidates in API response."
                )

            candidate = response.candidates[0]
            # FINISH_REASON_UNSPECIFIED = 0; STOP = 1; MAX_TOKENS = 2; SAFETY = 3; RECITATION = 4; OTHER = 5;
            if candidate.finish_reason != 1 : # Not STOP
                reason_map = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                reason_val = candidate.finish_reason
                reason_str = reason_map.get(reason_val, f"UNKNOWN ({reason_val})")
                
                error_detail_ui = f"Gemini API request finished atypically. Reason: {reason_str}."
                error_detail_dict = f"API request finished: {reason_str}."

                if candidate.safety_ratings:
                    ratings = "\n".join([f"  - {r.category}: {r.probability}" for r in candidate.safety_ratings])
                    error_detail_ui += f"\nSafety Ratings:\n{ratings}"
                    error_detail_dict += " Check safety ratings."
                
                # If content exists despite atypical finish, try to use it but warn
                if candidate.content and candidate.content.parts and any(p.text for p in candidate.content.parts if hasattr(p, 'text')):
                    st.warning(error_detail_ui + "\nAttempting to use partial content.")
                else: # No content and atypical finish reason
                    return self._return_error_structure(error_detail_ui, error_detail_dict)

            generated_text = ""
            if candidate.content and candidate.content.parts:
                generated_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text)
            
            if not generated_text.strip():
                 return self._return_error_structure(
                    "Gemini API returned empty text content, possibly due to safety filters or an issue with the prompt response.",
                    "Empty text content from API."
                 )

            # Clean the response to ensure it's valid JSON
            generated_text = generated_text.strip()
            if generated_text.startswith("```json"):
                generated_text = generated_text[7:]
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3]
            
            questions = json.loads(generated_text) # Can raise JSONDecodeError
            
            return {
                "technical": questions.get("technical", []),
                "behavioral": questions.get("behavioral", []),
                "project_specific": questions.get("project_specific", []),
                "scenario_based": questions.get("scenario_based", [])
            }
        
        except json.JSONDecodeError as je:
            return self._return_error_structure(
                f"Error decoding JSON from AI response: {je}. Check the raw response below.",
                f"Malformed JSON from API: {je}",
                # Pass raw text for debugging in UI if available
                # st.text_area("Problematic AI Response:", value=generated_text if 'generated_text' in locals() else "N/A", height=200)
            )
        except google.api_core.exceptions.GoogleAPIError as ge: # Specific Google API errors
             return self._return_error_structure(
                f"Gemini API Error: {ge.message} (Code: {ge.code})",
                f"Gemini API Error: {ge.message}"
             )
        except Exception as e: # Catch-all for other unexpected errors
            return self._return_error_structure(
                f"An unexpected error occurred during question generation: {e}",
                f"Unexpected error: {e}"
            )

    def generate_answer_for_question(self, question_text):
        if not self.client or not self.is_configured:
            return "Gemini client not configured. Cannot generate answer."
        if not question_text:
            return "No question provided to generate an answer."

        prompt = f"""
        You are an expert interviewer, career coach, and industry professional.
        A candidate has been asked the following interview question:
        ---
        Question: "{question_text}"
        ---
        Provide a model answer. (Instructions from previous prompt are still relevant)
        """ # Simplified prompt for brevity, original detailed prompt for answer generation is good.

        try:
            response = self.client.generate_content(prompt)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Answer generation blocked by Gemini. Reason: {response.prompt_feedback.block_reason}"
            if not response.candidates:
                return "Gemini returned no candidates for the answer."
            
            candidate = response.candidates[0]
            if candidate.finish_reason != 1: # Not STOP
                return f"Answer generation finished atypically. Reason: {candidate.finish_reason}"

            if candidate.content and candidate.content.parts:
                return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text).strip()
            return "No text content found in Gemini response for the answer."

        except Exception as e:
            st.error(f"Error generating answer with Gemini: {e}") # Log to streamlit console
            return f"Error generating answer: {str(e)}"

# --- Helper Function ---
def extract_text_from_pdf_bytes(pdf_bytes):
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
    return text

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="AI Interview Question Generator (Gemini)")

st.title("📄🤖 AI Interview Question Generator (Powered by Gemini)")
st.markdown("Upload your resume (PDF), optionally add a job description, and get AI-generated interview questions. Click on a question to get a sample answer!")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration (Gemini)")
    gemini_key_env = os.getenv("GEMINI_API_KEY")
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = gemini_key_env if gemini_key_env else ""
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key)

    if st.session_state.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
    
    st.session_state.model_type = "gemini"

    reinitialize_ai_gen = False
    if 'ai_question_gen' not in st.session_state or \
       st.session_state.ai_question_gen.gemini_api_key != os.getenv("GEMINI_API_KEY") or \
       not st.session_state.ai_question_gen.is_configured:
        reinitialize_ai_gen = True

    if reinitialize_ai_gen:
        if os.getenv("GEMINI_API_KEY"):
            with st.spinner("Initializing Gemini client..."):
                st.session_state.ai_question_gen = AIQuestionGenerator()
                if not st.session_state.ai_question_gen.is_configured:
                    # Warning already shown by AIQuestionGenerator constructor
                    pass
        elif 'ai_question_gen' in st.session_state:
            del st.session_state.ai_question_gen

    if 'ai_question_gen' in st.session_state and st.session_state.ai_question_gen.is_configured:
        st.sidebar.info(f"Using: Gemini ({st.session_state.ai_question_gen.model_name})")
    else:
        st.sidebar.warning("AI Generator not ready. Please provide your Gemini API key.")

# --- Main Area for Upload and Display ---
uploaded_file = st.file_uploader("1. Upload your Resume (PDF only)", type="pdf")

if uploaded_file:
    if 'resume_filename' not in st.session_state or st.session_state.resume_filename != uploaded_file.name:
        st.session_state.resume_filename = uploaded_file.name
        st.session_state.resume_bytes = uploaded_file.getvalue()
        st.session_state.resume_text = extract_text_from_pdf_bytes(st.session_state.resume_bytes)
        if 'generated_questions' in st.session_state: del st.session_state.generated_questions
        if 'current_question_to_answer' in st.session_state: del st.session_state.current_question_to_answer
        if 'current_answer' in st.session_state: del st.session_state.current_answer

    if st.session_state.resume_bytes:
        st.download_button("📥 Download Uploaded Resume", st.session_state.resume_bytes, st.session_state.resume_filename, "application/pdf")

    if st.session_state.resume_text:
        with st.expander("View Extracted Resume Text (first 1000 chars)"):
            st.text(st.session_state.resume_text[:1000] + "...")
    else:
        st.error("Could not extract text from the uploaded PDF.")

job_description = st.text_area("2. (Optional) Paste Job Description here:", height=150, key="job_desc_input")

if st.button("🧠 Generate Interview Questions", disabled=not uploaded_file or ('ai_question_gen' in st.session_state and not st.session_state.ai_question_gen.is_configured)):
    if 'ai_question_gen' not in st.session_state or not st.session_state.ai_question_gen.is_configured:
        st.error("Gemini AI Generator is not configured. Please check your API key in the sidebar.")
    elif st.session_state.resume_text:
        with st.spinner("Generating questions with Gemini... This may take a moment."):
            questions_data = st.session_state.ai_question_gen.generate_questions_from_text(
                st.session_state.resume_text,
                job_description
            )
            st.session_state.generated_questions = questions_data
            if 'current_question_to_answer' in st.session_state: del st.session_state.current_question_to_answer
            if 'current_answer' in st.session_state: del st.session_state.current_answer

            # Conditional success message
            is_error_in_response = False
            if questions_data and questions_data.get("technical") and \
               isinstance(questions_data["technical"], list) and \
               questions_data["technical"] and \
               questions_data["technical"][0].startswith("Error:"):
                is_error_in_response = True
            
            has_any_actual_questions = any(
                isinstance(q_list, list) and q_list and not (isinstance(q_list[0], str) and q_list[0].startswith("Error:"))
                for q_list in questions_data.values() if q_list # check only non-empty lists
            )


            if not is_error_in_response and has_any_actual_questions:
                st.success("Questions generated!")
            # Error messages are displayed by AIQuestionGenerator directly using st.error

    else:
        st.warning("Please upload a resume first.")

if 'generated_questions' in st.session_state and st.session_state.generated_questions:
    st.subheader("🎯 Generated Interview Questions:")
    q_data = st.session_state.generated_questions
    categories = {
        "technical": "💻 Technical Deep Dive",
        "project_specific": "🛠️ Project-Specific",
        "behavioral": "🤝 Behavioral & Situational",
        "scenario_based": "🧠 Scenario-Based/Problem-Solving"
    }

    for cat_key, cat_name in categories.items():
        if q_data.get(cat_key):
            st.markdown(f"#### {cat_name}")
            # Check if the only content in this category is an error message
            if isinstance(q_data[cat_key], list) and len(q_data[cat_key]) == 1 and \
               isinstance(q_data[cat_key][0], str) and q_data[cat_key][0].startswith("Error:"):
                # Error already displayed by AIQuestionGenerator, or it's the main error display for the category.
                # We can choose to re-display it or assume it was shown.
                # For now, let's show it if it's the only item, to make it clear.
                st.warning(q_data[cat_key][0]) # Use warning for errors displayed as questions
            else:
                for i, question in enumerate(q_data[cat_key]):
                    if isinstance(question, str) and question.startswith("Error:"):
                        st.warning(question) # Display error messages differently if they appear within a list of questions
                        continue

                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.markdown(f"{i+1}. {question}")
                    with col2:
                        button_key = f"answer_btn_{cat_key}_{i}"
                        if st.button("💡 Get Answer", key=button_key):
                            st.session_state.current_question_to_answer = question
                            st.session_state.current_answer = "loading"
                    
                    if 'current_question_to_answer' in st.session_state and st.session_state.current_question_to_answer == question:
                        if st.session_state.current_answer == "loading":
                            with st.spinner("Generating answer with Gemini..."):
                                answer = st.session_state.ai_question_gen.generate_answer_for_question(question)
                                st.session_state.current_answer = answer
                        
                        if st.session_state.current_answer and st.session_state.current_answer != "loading":
                            with st.expander(f"🤖 Model Answer for: \"{question[:50]}...\"", expanded=True):
                                st.markdown(st.session_state.current_answer)
            st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Google Gemini")