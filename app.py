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

# --- Helper Function: API Call (for Rankings) ---

def fetch_ai_scores(universities, is_international, scholarships):
    """
    Calls the Gemini API to get objective scores for universities.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("API key not found. Please add it to your Streamlit Secrets.")
        return None

    system_prompt = (
        "You are an expert college admissions and data analyst. "
        "Your job is to provide objective scores (from 1 to 10, where 1 is worst and 10 is best) "
        "for a list of universities based on specific factors. "
        "For 'Total Cost (Lower is Better)', a lower cost MUST result in a higher score (e.g., the cheapest university gets a 10). "
        "For 'Work Authorization (OPT/STEM OPT)', a 3-year STEM OPT-eligible program is a 10, a 1-year OPT is a 5, and no authorization is a 1. "
        "When scoring 'Total Cost', you MUST first deduct any scholarship amount provided for that university before making your 1-10 score comparison. "
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
    
    if is_international:
        user_prompt += "\n\nIMPORTANT: For all 'cost' calculations, you MUST use the international or out-of-state tuition and fees."
    
    # Create a list of scholarships to add to the prompt
    # Use the index 'i' in the scholarship dict to handle duplicate names
    scholarship_list = [f"- {uni}: ${amount}" for uni, amount in scholarships.items() if amount > 0]
    if scholarship_list:
        scholarship_str = "\n".join(scholarship_list)
        user_prompt += f"\n\nIMPORTANT: You MUST deduct these scholarship amounts from the total cost before calculating the 'cost' score:\n{scholarship_str}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}],
    }

    try:
        response = requests.post(
            API_URL + api_key,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=120 
        )
        
        if response.status_code != 200:
            st.error(f"Error from API: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        
        content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
        
        start_index = content.find('[')
        end_index = content.rfind(']')
        
        if start_index != -1 and end_index != -1:
            json_string = content[start_index:end_index+1]
            try:
                return json.loads(json_string) 
            except json.JSONDecodeError as e:
                st.error(f"Error decoding the AI's JSON response: {e}")
                st.error(f"AI returned: {content}")
                return None
        else:
            st.error(f"AI did not return valid JSON. AI returned: {content}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network error while calling API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Helper Function: API Call (for Chat) ---

def fetch_chat_response(chat_history, results_context):
    """
    Calls the Gemini API to get a chat response based on the results.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("API key not found. Please add it to your Streamlit Secrets.")
        return "Sorry, I can't connect to my brain right now. Please check the API key."

    system_prompt = (
        "You are a helpful college admissions analyst. "
        "Your job is to answer follow-up questions about a set of college decision results that I will provide. "
        "Be concise, friendly, and helpful. Use the data I provide to justify your answers."
    )
    
    api_history = []
    
    context_prompt = (
        f"Here are the college decision results we are discussing. "
        f"Use this data to answer my questions. Do not show this table to me again, just use it as context. "
        f"The data is a JSON array: {results_context}"
    )
    api_history.append({"role": "user", "parts": [{"text": context_prompt}]})
    api_history.append({"role": "model", "parts": [{"text": "Understood. I have the results table. What's your question about it?"}]})
    
    for msg in chat_history:
        api_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["content"]}]
        })

    payload = {
        "contents": api_history,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    try:
        response = requests.post(
            API_URL + api_key,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=60
        )
        
        if response.status_code != 200:
            return f"Error from API: {response.status_code} - {response.text}"
        
        result = response.json()
        
        content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Sorry, I had trouble thinking of a response.')
        return content

    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Main App ---

# Initialize session state
if 'universities' not in st.session_state:
    st.session_state.universities = ["", ""]
if 'weights' not in st.session_state:
    st.session_state.weights = {f['id']: 100 // len(ALL_FACTORS) for f in ALL_FACTORS}
    st.session_state.weights[ALL_FACTORS[-1]['id']] += 100 % len(ALL_FACTORS)
if 'user_scores' not in st.session_state:
    st.session_state.user_scores = {}
if 'ai_scores' not in st.session_state:
    st.session_state.ai_scores = None
if 'calculations' not in st.session_state:
    st.session_state.calculations = None
if 'is_international' not in st.session_state:
    st.session_state.is_international = False
if 'scholarships' not in st.session_state:
    st.session_state.scholarships = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Sidebar (Controls) ---
st.sidebar.title("🎓 College Matrix Controls")

# 1. Universities
with st.sidebar.expander("1. Enter Universities", expanded=True):
    # Use a new list in session state to hold the university inputs
    if 'university_inputs' not in st.session_state:
        st.session_state.university_inputs = ["", ""]
        
    # We iterate over the *indices* of the list
    for i in range(len(st.session_state.university_inputs)):
        col1, col2 = st.columns([4, 1])
        st.session_state.university_inputs[i] = col1.text_input(
            f"University {i + 1}", 
            st.session_state.university_inputs[i], 
            label_visibility="collapsed",
            placeholder=f"University {i + 1}",
            key=f"uni_input_{i}" # Use index for unique key
        )
        if col2.button("X", key=f"remove_uni_{i}", help="Remove university") and len(st.session_state.university_inputs) > 2:
            st.session_state.university_inputs.pop(i)
            st.rerun()

    if st.button("Add University", use_container_width=True) and len(st.session_state.university_inputs) < 5:
        st.session_state.university_inputs.append("")
        st.rerun()
    
    # valid_universities is now built from university_inputs
    valid_universities = [u.strip() for u in st.session_state.university_inputs if u.strip()]
    
    st.session_state.is_international = st.checkbox(
        "I am an international / out-of-state student",
        value=st.session_state.is_international
    )

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

# 3. Scholarships
with st.sidebar.expander("3. Enter Scholarships ($)", expanded=True):
    if not valid_universities:
        st.info("Add universities above to enter scholarships.")
    else:
        # --- KEY FIX ---
        # We iterate over the *index* and *name* of the valid list
        # This creates unique keys even if names are duplicated
        st.session_state.scholarships = {} # Clear scholarships to rebuild
        for i, uni_name in enumerate(valid_universities):
            st.session_state.scholarships[f"{uni_name}_{i}"] = st.number_input(
                f"Scholarship for {uni_name} ($)",
                min_value=0,
                step=1000,
                key=f"scholarship_{i}" # Use index for unique key
            )

# 4. Manual Scores
with st.sidebar.expander("4. Enter Your Personal Scores (1-10)", expanded=True):
    if not valid_universities:
        st.info("Add universities above to enter your scores.")
    else:
        # --- KEY FIX ---
        # We iterate over the *index* and *name*
        st.session_state.user_scores = {} # Clear scores to rebuild
        for i, uni_name in enumerate(valid_universities):
            st.markdown(f"**{uni_name}** (Entry {i+1})")
            st.session_state.user_scores[f"{uni_name}_{i}"] = {}
            for factor in USER_SCORED_FACTORS:
                st.session_state.user_scores[f"{uni_name}_{i}"][factor['id']] = st.number_input(
                    factor['name'], 1, 10, 5, 
                    key=f"score_{i}_{factor['id']}" # Use index for unique key
                )

# 5. Generate Button
if st.sidebar.button("Generate AI Rankings", type="primary", use_container_width=True, disabled=len(valid_universities) < 2):
    with st.spinner("AI is researching and ranking your universities... This may take a moment."):
        st.session_state.ai_scores = None
        st.session_state.calculations = None
        
        # Pass the valid university names, but the indexed scholarship dict
        raw_ai_scores_list = fetch_ai_scores(
            valid_universities, # Send the list of names
            st.session_state.is_international,
            st.session_state.scholarships # Send the dict with {uni_name_i: amount}
        )
        
        if raw_ai_scores_list:
            raw_ai_scores_dict = {}
            for item in raw_ai_scores_list:
                name = item.get('university_name')
                if name:
                    raw_ai_scores_dict[name] = item
            
            # This dict will hold the final, matched scores
            normalized_ai_scores = {}
            # We must iterate by index to match scores to the right input box
            for i, uni_name in enumerate(valid_universities):
                lower_uni_name = uni_name.lower()
                found_key = next((key for key in raw_ai_scores_dict if lower_uni_name in key.lower()), None)
                
                # The key for this score is uni_name + index
                score_key = f"{uni_name}_{i}" 
                
                if found_key:
                    normalized_ai_scores[score_key] = raw_ai_scores_dict[found_key]
                else:
                    st.warning(f"AI did not return data for '{uni_name}'. Scores will be 0.")
                    normalized_ai_scores[score_key] = {f['id']: 0 for f in AI_SCORED_FACTORS}
            
            st.session_state.ai_scores = normalized_ai_scores

            # --- Perform Calculations ---
            scores_data = []
            table_data = []
            total_w = sum(st.session_state.weights.values()) or 1
            
            # Iterate by index to match all our data
            for i, uni_name in enumerate(valid_universities):
                score_key = f"{uni_name}_{i}"
                weighted_score = 0
                row = {"University": f"{uni_name} (Entry {i+1})"}
                
                for factor in ALL_FACTORS:
                    fid = factor['id']
                    score = 0
                    if fid in [f['id'] for f in AI_SCORED_FACTORS]:
                        # Get score from the AI dict using the indexed key
                        score = st.session_state.ai_scores.get(score_key, {}).get(fid, 0)
                    else:
                        # Get score from the user dict using the indexed key
                        score = st.session_state.user_scores.get(score_key, {}).get(fid, 0)
                    
                    weight = st.session_state.weights[fid] / total_w
                    weighted_score += score * weight
                    row[factor['name']] = score
                
                final_score = round(weighted_score * 10, 1) # Scale to 1-100
                scores_data.append({'name': f"{uni_name} (Entry {i+1})", 'score': final_score})
                row['Final Score'] = final_score
                table_data.append(row)
            
            winner = max(scores_data, key=lambda x: x['score'])
            
            st.session_state.calculations = {
                'scores': scores_data,
                'table': table_data, 
                'winner': winner
            }
            
            st.session_state.messages = [] # Clear old chat on new results
            st.success("Analysis Complete!")
        else:
            st.error("Failed to get AI scores. Please check the error messages.")

# --- Main Page (Results) ---
st.title("🎓 AI-Powered University Decision Matrix")
st.write("Compare universities by weighting what matters to you. Let AI find the objective data.")

if not st.session_state.calculations:
    st.info("Fill in the details on the left and click 'Generate AI Rankings' to see your results.")
    # --- IMAGE FIX: Added https:// ---
    st.image("https://placehold.co/1200x600/FAFAFA/CCCCCC?text=Your+Results+Will+Appear+Here", use_container_width=True)
else:
    calc = st.session_state.calculations
    
    # 1. Recommendation
    st.success(f"**Recommendation: {calc['winner']['name']}**")
    st.markdown(f"Based on your weights, **{calc['winner']['name']}** is the best fit with a score of **{calc['winner']['score']}**.")

    # 2. Comparison Chart
    st.subheader("Final Score Comparison")
    
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
    
    df = pd.DataFrame(calc['table'])
    df = df.set_index("University")
    
    def style_ai_columns(col_name):
        is_ai_col = any(col_name == f['name'] for f in AI_SCORED_FACTORS)
        return 'background-color: #EFF6FF' if is_ai_col else None
        
    # --- WARNING FIX: Replaced applymap_index with map_index ---
    st.dataframe(
        df.style.map_index(style_ai_columns, axis=1)
                .apply(lambda x: ['background-color: #DBEAFE' if x.name == 'Final Score' else '' for i in x], axis=0)
                .format("{:.1f}", subset=[col for col in df.columns if col != "University"])
    )
    st.caption("Blue-tinted columns are scored by AI. White columns are your manual scores.")
    
    # --- Chat Interface ---
    st.divider()
    st.subheader("Ask About Your Results")
    
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Get new chat input from user
    if prompt := st.chat_input("e.g., Why did Harvard score so low on cost?"):
        # Add user's message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user's message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the chat history and the results table (as JSON) to the AI
                results_context = json.dumps(calc['table'])
                response = fetch_chat_response(st.session_state.messages, results_context)
                
                # Display AI's response
                st.markdown(response)
                
                # Add AI's response to history
                st.session_state.messages.append({"role": "assistant", "content": response})