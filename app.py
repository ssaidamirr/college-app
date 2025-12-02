import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json

# --- Constants ---

# Factors the AI will score (1-10, 1=worst, 10=best)
AI_SCORED_FACTORS = [
    {'id': 'cost', 'name': 'Total Cost (Lower is Better)'},
    {'id': 'opt', 'name': 'Work Authorization (OPT/STEM OPT)'},
    {'id': 'careers', 'name': 'Career Opportunities'},
    {'id': 'prestige', 'name': 'Academic Prestige'},
    {'id': 'stress', 'name': 'Stress/Workload'},
    {'id': 'living', 'name': 'Living Environment/Life'},
]

# Factors the user must score manually (1-10)
USER_SCORED_FACTORS = [
    {'id': 'fit', 'name': 'Program Fit (my goals)'},
    {'id': 'feel', 'name': 'Personal Feelings Towards the Uni'},
]

ALL_FACTORS = AI_SCORED_FACTORS + USER_SCORED_FACTORS

# Gemini API Model
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key="

# --- Page Configuration ---
st.set_page_config(
    page_title="AI College Decision Matrix",
    page_icon="🎓",
    layout="wide"
)

# --- Helper Function: API Call ---

def fetch_ai_scores(universities):
    """
    Calls the Gemini API to get objective scores for universities.
    This function runs securely on the server.
    """
    try:
        # Get the API key from Streamlit Secrets
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("API key not found. Please add it to your Streamlit Secrets.")
        return None

    # --- Prompt Fix: Be very strict ---
    system_prompt = (
        "You are an expert college admissions and data analyst. "
        "Your job is to provide objective scores (from 1 to 10, where 1 is worst and 10 is best) "
        "for a list of universities based on specific factors. "
        "For 'Total Cost (Lower is Better)', a lower cost MUST result in a higher score (e.g., the cheapest university gets a 10). "
        "For 'Work Authorization (OPT/STEM OPT)', a 3-year STEM OPT-eligible program is a 10, a 1-year OPT is a 5, and no authorization is a 1. "
        "You MUST respond ONLY with a valid JSON array of objects. "
        "Do NOT include any other text, markdown formatting, greetings, or explanations. "
        "Your entire response must be ONLY the JSON array."
    )
    
    uni_list_str = ', '.join([f'"{u}"' for u in universities])
    factors_list_str = '\n- '.join([f['id'] for f in AI_SCORED_FACTORS]) # Use IDs for keys
    
    user_prompt = (
        f"For the following list of universities: [{uni_list_str}], "
        "provide scores (1-10) for these factors. "
        "Return your response ONLY as a JSON array, with one object per university. "
        "Each object MUST have a 'university_name' key and keys for each factor ID. "
        f"The factor ID keys are: {factors_list_str}"
    )
    
    # --- API Call Fix: Remove the generationConfig schema ---
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}],
        # We removed the 'generationConfig' with the schema, which caused the 400 error.
        # We now rely on the strong prompt to get JSON back.
    }

    try:
        response = requests.post(
            API_URL + api_key,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=120 # 2 minute timeout for complex searches
        )
        
        if response.status_code != 200:
            st.error(f"Error from API: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        
        # --- Parsing Fix: Find the JSON in the plain text response ---
        content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
        
        # The AI might return just the JSON, or it might wrap it in text or markdown.
        # We need to find the start of the list '[' and the end ']'
        start_index = content.find('[')
        end_index = content.rfind(']')
        
        if start_index != -1 and end_index != -1:
            json_string = content[start_index:end_index+1]
            try:
                return json.loads(json_string) # The API now returns a list (array)
            except json.JSONDecodeError as e:
                st.error(f"Error decoding the AI's JSON response: {e}")
                st.error(f"AI returned: {content}")
                return None
        else:
            st.error(f"AI did not return valid JSON. AI returned: {content}")
            return None
        # --- End of Parsing Fix ---

    except requests.exceptions.RequestException as e:
        st.error(f"Network error while calling API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Main App ---

# Initialize session state
if 'universities' not in st.session_state:
    st.session_state.universities = ["", ""]
if 'weights' not in st.session_state:
    st.session_state.weights = {f['id']: 100 // len(ALL_FACTORS) for f in ALL_FACTORS}
    # Adjust last one to sum to 100
    st.session_state.weights[ALL_FACTORS[-1]['id']] += 100 % len(ALL_FACTORS)
if 'user_scores' not in st.session_state:
    st.session_state.user_scores = {}
if 'ai_scores' not in st.session_state:
    st.session_state.ai_scores = None
if 'calculations' not in st.session_state:
    st.session_state.calculations = None

# --- Sidebar (Controls) ---
st.sidebar.title("🎓 College Matrix Controls")

# 1. Universities
with st.sidebar.expander("1. Enter Universities", expanded=True):
    for i in range(len(st.session_state.universities)):
        col1, col2 = st.columns([4, 1])
        st.session_state.universities[i] = col1.text_input(
            f"University {i + 1}", 
            st.session_state.universities[i], 
            label_visibility="collapsed",
            placeholder=f"University {i + 1}"
        )
        if col2.button("X", key=f"remove_uni_{i}", help="Remove university") and len(st.session_state.universities) > 2:
            st.session_state.universities.pop(i)
            # Rerun to update the UI immediately
            st.rerun()

    if st.button("Add University", use_container_width=True) and len(st.session_state.universities) < 5:
        st.session_state.universities.append("")
        st.rerun()

valid_universities = [u.strip() for u in st.session_state.universities if u.strip()]

# 2. Weights
with st.sidebar.expander("2. Set Factor Weights", expanded=True):
    for factor in ALL_FACTORS:
        st.session_state.weights[factor['id']] = st.slider(
            factor['name'], 0, 100, st.session_state.weights[factor['id']], 5
        )
    
    total_weight = sum(st.session_state.weights.values())
    if total_weight == 100:
        st.success(f"Total Weight: {total_weight}%")
    else:
        st.warning(f"Total Weight: {total_weight}% (Will be normalized)")

# 3. Manual Scores
with st.sidebar.expander("3. Enter Your Scores (1-10)", expanded=True):
    if not valid_universities:
        st.info("Add universities above to enter your scores.")
    else:
        # Initialize user_scores dictionary if needed
        for uni in valid_universities:
            if uni not in st.session_state.user_scores:
                st.session_state.user_scores[uni] = {}

        for uni in valid_universities:
            st.markdown(f"**{uni}**")
            for factor in USER_SCORED_FACTORS:
                st.session_state.user_scores[uni][factor['id']] = st.number_input(
                    factor['name'], 1, 10, 
                    st.session_state.user_scores[uni].get(factor['id'], 5), 
                    key=f"score_{uni}_{factor['id']}"
                )

# 4. Generate Button
if st.sidebar.button("Generate AI Rankings", type="primary", use_container_width=True, disabled=len(valid_universities) < 2):
    with st.spinner("AI is researching and ranking your universities... This may take a moment."):
        st.session_state.ai_scores = None
        st.session_state.calculations = None
        
        # This will now be a LIST of objects, e.g., [{'university_name': 'Harvard', ...}]
        raw_ai_scores_list = fetch_ai_scores(valid_universities)
        
        if raw_ai_scores_list:
            
            # --- Data Processing Fix ---
            # Convert the list into a dictionary for easier lookup
            raw_ai_scores_dict = {}
            for item in raw_ai_scores_list:
                name = item.get('university_name')
                if name:
                    raw_ai_scores_dict[name] = item
            
            # Normalize AI scores (match keys from the dict we just made)
            normalized_ai_scores = {}
            for uni_name in valid_universities:
                lower_uni_name = uni_name.lower()
                # Find the key in raw_ai_scores_dict that matches
                found_key = next((key for key in raw_ai_scores_dict if lower_uni_name in key.lower()), None)
                
                if found_key:
                    normalized_ai_scores[uni_name] = raw_ai_scores_dict[found_key]
                else:
                    st.warning(f"AI did not return data for '{uni_name}'. Scores will be 0.")
                    normalized_ai_scores[uni_name] = {f['id']: 0 for f in AI_SCORED_FACTORS}
            
            st.session_state.ai_scores = normalized_ai_scores
            # --- End of Data Processing Fix ---

            # --- Perform Calculations ---
            scores_data = []
            table_data = []
            total_w = sum(st.session_state.weights.values()) or 1
            
            for uni in valid_universities:
                weighted_score = 0
                row = {"University": uni}
                
                for factor in ALL_FACTORS:
                    fid = factor['id']
                    score = 0
                    if fid in [f['id'] for f in AI_SCORED_FACTORS]:
                        # Get the score from the normalized dictionary
                        score = st.session_state.ai_scores.get(uni, {}).get(fid, 0)
                    else:
                        score = st.session_state.user_scores.get(uni, {}).get(fid, 0)
                    
                    weight = st.session_state.weights[fid] / total_w
                    weighted_score += score * weight
                    row[factor['name']] = score
                
                final_score = round(weighted_score * 10, 1) # Scale to 1-100
                scores_data.append({'name': uni, 'score': final_score})
                row['Final Score'] = final_score
                table_data.append(row)
            
            winner = max(scores_data, key=lambda x: x['score'])
            st.session_state.calculations = {
                'scores': scores_data,
                'table': table_data,
                'winner': winner
            }
            st.success("Analysis Complete!")
        else:
            st.error("Failed to get AI scores. Please check the error messages.")

# --- Main Page (Results) ---
st.title("🎓 AI-Powered University Decision Matrix")
st.write("Compare universities by weighting what matters to you. Let AI find the objective data.")

if not st.session_state.calculations:
    st.info("Fill in the details on the left and click 'Generate AI Rankings' to see your results.")
    st.image("https://placehold.co/1200x600/FAFAFA/CCCCCC?text=Your+Results+Will+Appear+Here", use_column_width=True)
else:
    calc = st.session_state.calculations
    
    # 1. Recommendation
    st.success(f"**Recommendation: {calc['winner']['name']}**")
    st.markdown(f"Based on your weights, **{calc['winner']['name']}** is the best fit with a score of **{calc['winner']['score']}**.")

    # 2. Comparison Chart
    st.subheader("Final Score Comparison")
    
    # Create Plotly bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=[item['name'] for item in calc['scores']],
                y=[item['score'] for item in calc['scores']],
                text=[item['score'] for item in calc['scores']],
                textposition='auto',
                marker_color='#3B82F6'
            )
        ]
    )
    fig.update_layout(
        title="Final Weighted Scores (out of 100)",
        xaxis_title="University",
        yaxis_title="Final Score",
        yaxis_range=[0, 100],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Detailed Table
    st.subheader("Detailed Score Breakdown")
    
    # Convert table data to DataFrame for better display
    df = pd.DataFrame(calc['table'])
    df = df.set_index("University")
    
    # Highlight AI-scored columns
    def style_ai_columns(col_name):
        is_ai_col = any(col_name == f['name'] for f in AI_SCORED_FACTORS)
        return 'background-color: #EFF6FF' if is_ai_col else None
        
    st.dataframe(
        df.style.applymap_index(style_ai_columns, axis=1)
                .apply(lambda x: ['background-color: #DBEAFE' if x.name == 'Final Score' else '' for i in x], axis=0)
                .format("{:.1f}", subset=[col for col in df.columns if col != "University"])
    )
    st.caption("Blue-tinted columns are scored by AI. White columns are your manual scores.")