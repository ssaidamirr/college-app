import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import re

# --- Constants ---

AI_SCORED_FACTORS = [
    {'id': 'cost', 'name': 'Annual Net Cost (Lower is Better)'},
    {'id': 'opt', 'name': 'Work Authorization (based on Major)'},
    {'id': 'careers', 'name': 'Career Opportunities (based on Major)'},
    {'id': 'prestige', 'name': 'Academic Prestige (for Major)'},
    {'id': 'stress', 'name': 'Stress/Workload'},
    {'id': 'living', 'name': 'Living Environment/Life'},
]

USER_SCORED_FACTORS = [
    {'id': 'fit', 'name': 'Program Fit (my goals)'},
    {'id': 'feel', 'name': 'Personal Feelings Towards the Uni'},
]

ALL_FACTORS = AI_SCORED_FACTORS + USER_SCORED_FACTORS

GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key="

# --- Helper Function to Parse Dollar Strings ---
def parse_dollars(s):
    if isinstance(s, (int, float)):
        return s
    
    s = str(s).lower().strip()
    s = re.sub(r"[\$,/yr]", "", s)  # Remove $, commas, /yr
    
    if 'k' in s:
        s = s.replace('k', '')
        multiplier = 1000
    elif 'm' in s:
        s = s.replace('m', '')
        multiplier = 1000000
    else:
        multiplier = 1
        
    try:
        return float(s) * multiplier
    except ValueError:
        return 0

# --- Page Configuration ---
st.set_page_config(
    page_title="AI College Decision Matrix",
    page_icon="🎓",
    layout="wide"
)

# --- Helper Function: API Call (for Rankings) ---

def fetch_ai_scores(universities_with_majors, is_international, scholarships, degree_level, opt_is_important):
    """
    Calls the Gemini API to get objective scores for universities.
    
    NEW: Only requests 'opt' data if it's important to the user.
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("API key not found. Please add it to your Streamlit Secrets.")
        return None

    # --- DYNAMIC PROMPT ---
    # Create the list of factors to score
    active_ai_factors = []
    for factor in AI_SCORED_FACTORS:
        if factor['id'] == 'opt' and not opt_is_important:
            continue
        active_ai_factors.append(factor)
    
    factors_list_str = '\n- '.join([f['id'] for f in active_ai_factors])
    # --- END DYNAMIC PROMPT ---

    system_prompt = (
        "You are an expert college admissions and career analyst. "
        "Your job is to provide objective scores (from 1 to 10, 1=worst, 10=best) "
        "and raw data for a list of university/major pairs. "
        "You MUST respond ONLY with a valid JSON array of objects. "
        "Do NOT include any other text, markdown formatting, or explanations. "
        "\n\nSCORING RULES:"
        "1. 'cost' (Annual Net Cost): Find the **annual (per-year)** total cost of attendance (tuition, fees, room, board). "
        "   The scholarship amount provided by the user is also *per year*. You MUST subtract this from the *annual* total cost to get the *annual net cost*. "
        "   Score the 'cost' factor based on this final *annual net cost*. A lower net cost MUST get a higher score."
        "   If the user is international/out-of-state, you MUST use that specific tuition rate."
        "2. 'opt' (Work Authorization): (If requested) Based on the *specific major* and *degree level*, determine its U.S. work authorization. "
        "   3-year STEM OPT eligible gets a 10. 1-year standard OPT gets a 5. No OPT gets a 1."
        "3. 'careers' & 'prestige': Scores should be specific to the *major* at that university."
        "4. 'note': For each factor, you MUST provide a 'note' with the raw data used (e.g., '$60k/yr net cost', '3-year STEM OPT eligible')."
        "5. RAW DATA: You MUST also return 'estimated_annual_cost' (the *annual* total cost as a string, e.g., '$60,000/yr') and 'estimated_starting_salary' (avg. starting salary for that major as a string, e.g., '$90,000/yr')."
    )
    
    uni_major_list_str = json.dumps(universities_with_majors)
    
    user_prompt = (
        f"I am looking at {degree_level} degrees. "
        f"For the following list of university/major pairs: {uni_major_list_str}, "
        "provide scores and raw data. "
        "Return your response ONLY as a JSON array, with one object per university. "
        "Each object MUST have: "
        " 1. 'university_name' (string) "
        " 2. 'estimated_annual_cost' (string, 1-year total cost) "
        " 3. 'estimated_starting_salary' (string, avg starting salary for major) "
        " 4. An object for each factor ID I requested, containing 'score' (1-10) and 'note' (string). "
        f"The factor ID keys are: {factors_list_str}"
    )
    
    if is_international:
        user_prompt += "\n\nIMPORTANT: I am an international / out-of-state student. You MUST use the international/out-of-state total cost of attendance for the 'cost' score."
    
    scholarship_prompt_list = []
    for i, uni in enumerate(universities_with_majors):
        scholarship_key = f"{uni['name']}_{i}"
        amount = scholarships.get(scholarship_key, 0)
        if amount > 0:
            scholarship_prompt_list.append(f"- {uni['name']} ({uni['major']}): ${amount} per year")
            
    if scholarship_prompt_list:
        scholarship_str = "\n".join(scholarship_prompt_list)
        user_prompt += f"\n\nIMPORTANT: You MUST deduct these annual scholarship amounts from the annual total cost before calculating the 'cost' score:\n{scholarship_str}"

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
            timeout=180 
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
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("API key not found. Please add it to your Streamlit Secrets.")
        return "Sorry, I can't connect to my brain right now. Please check the API key."

    system_prompt = (
        "You are a helpful college admissions and career analyst. "
        "Your job is to answer follow-up questions about a set of college decision results that I will provide. "
        "Be concise, friendly, and helpful. Use the data I provide to justify your answers. "
        "The user's results table contains scores (e.g., '5/10 (note)'), a Final Score, "
        "and new columns: 'Est. Salary', 'Est. Annual Cost', and 'Estimated ROI'. "
        "When asked 'how much was X', use the 'Est. Annual Cost' column or the 'note' in the 'Annual Net Cost' cell. "
        "When asked about salary or ROI, use the new columns."
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
if 'degree_level' not in st.session_state:
    st.session_state.degree_level = "Bachelor's"
if 'university_inputs' not in st.session_state:
    st.session_state.university_inputs = [{'name': '', 'major': ''}, {'name': '', 'major': ''}]
if 'opt_is_important' not in st.session_state:
    st.session_state.opt_is_important = True # Default for international

# --- Sidebar (Controls) ---
st.sidebar.title("🎓 College Matrix Controls")

# 1. Universities
with st.sidebar.expander("1. Enter Universities & Degree", expanded=True):
    
    st.session_state.degree_level = st.radio(
        "Select Degree Level",
        ("Bachelor's", "Master's"),
        index=0 if st.session_state.degree_level == "Bachelor's" else 1,
        horizontal=True
    )
    
    st.divider()
    
    for i in range(len(st.session_state.university_inputs)):
        st.markdown(f"**University {i + 1}**")
        col1, col2 = st.columns([3, 2])
        st.session_state.university_inputs[i]['name'] = col1.text_input(
            "University Name", 
            st.session_state.university_inputs[i]['name'], 
            label_visibility="collapsed",
            placeholder="University Name",
            key=f"uni_name_{i}"
        )
        st.session_state.university_inputs[i]['major'] = col2.text_input(
            "Major", 
            st.session_state.university_inputs[i]['major'], 
            label_visibility="collapsed",
            placeholder="Major (e.g., Computer Science)",
            key=f"uni_major_{i}"
        )
        
        if st.button(f"Remove Entry {i + 1}", key=f"remove_uni_{i}", help="Remove university") and len(st.session_state.university_inputs) > 2:
            st.session_state.university_inputs.pop(i)
            st.rerun()
        
        st.divider()

    if st.button("Add University", use_container_width=True) and len(st.session_state.university_inputs) < 5:
        st.session_state.university_inputs.append({'name': '', 'major': ''})
        st.rerun()
    
    valid_universities = [
        u for u in st.session_state.university_inputs 
        if u['name'].strip() and u['major'].strip()
    ]
    
    st.session_state.is_international = st.checkbox(
        "I am an international / out-of-state student",
        value=st.session_state.is_international
    )

# 2. Weights
with st.sidebar.expander("2. Set Factor Weights", expanded=True):
    
    # --- NEW: OPT Checkbox ---
    st.session_state.opt_is_important = st.checkbox(
        "OPT/Work Authorization is important to me",
        value=st.session_state.opt_is_important
    )
    st.caption("Uncheck this if you are a domestic student.")
    # --- END NEW ---
    
    active_factors = []
    for factor in ALL_FACTORS:
        # Conditionally add the 'opt' factor
        if factor['id'] == 'opt':
            if st.session_state.opt_is_important:
                active_factors.append(factor)
                st.session_state.weights[factor['id']] = st.slider(
                    factor['name'], 0, 100, st.session_state.weights[factor['id']], 5
                )
            else:
                # If not important, set its weight to 0
                st.session_state.weights[factor['id']] = 0
        else:
            # Add all other factors
            active_factors.append(factor)
            st.session_state.weights[factor['id']] = st.slider(
                factor['name'], 0, 100, st.session_state.weights[factor['id']], 5
            )
    
    # Calculate total weight based on *active* factors only
    total_weight = sum(st.session_state.weights[f['id']] for f in active_factors)
    
    if total_weight == 100:
        st.success(f"Total Weight: {total_weight}%")
    else:
        st.warning(f"Total Weight: {total_weight}% (Weights will be normalized to 100%)")

# 3. Scholarships
with st.sidebar.expander("3. Enter Scholarships (Annual) ($)", expanded=True):
    if not valid_universities:
        st.info("Add universities & majors above to enter scholarships.")
    else:
        st.session_state.scholarships = {}
        for i, uni in enumerate(valid_universities):
            st.session_state.scholarships[f"{uni['name']}_{i}"] = st.number_input(
                f"Scholarship (per year) for {uni['name']} ($)",
                min_value=0,
                step=1000,
                key=f"scholarship_{i}"
            )

# 4. Manual Scores
with st.sidebar.expander("4. Enter Your Personal Scores (1-10)", expanded=True):
    if not valid_universities:
        st.info("Add universities & majors above to enter your scores.")
    else:
        st.session_state.user_scores = {}
        for i, uni in enumerate(valid_universities):
            st.markdown(f"**{uni['name']}** ({uni['major']})")
            st.session_state.user_scores[f"{uni['name']}_{i}"] = {}
            for factor in USER_SCORED_FACTORS:
                st.session_state.user_scores[f"{uni['name']}_{i}"][factor['id']] = st.number_input(
                    factor['name'], 1, 10, 5, 
                    key=f"score_{i}_{factor['id']}"
                )

# 5. Generate Button
if st.sidebar.button("Generate AI Rankings", type="primary", use_container_width=True, disabled=len(valid_universities) < 2):
    with st.spinner("AI is researching and ranking your universities... This is a complex task and may take a moment."):
        st.session_state.ai_scores = None
        st.session_state.calculations = None
        
        raw_ai_scores_list = fetch_ai_scores(
            valid_universities,
            st.session_state.is_international,
            st.session_state.scholarships,
            st.session_state.degree_level,
            st.session_state.opt_is_important # Pass the new flag
        )
        
        if raw_ai_scores_list:
            raw_ai_scores_dict = {}
            for item in raw_ai_scores_list:
                name = item.get('university_name')
                if name:
                    raw_ai_scores_dict[name] = item
            
            normalized_ai_scores = {}
            for i, uni in enumerate(valid_universities):
                uni_name = uni['name']
                lower_uni_name = uni_name.lower()
                found_key = next((key for key in raw_ai_scores_dict if lower_uni_name in key.lower()), None)
                
                score_key = f"{uni_name}_{i}" 
                
                if found_key:
                    normalized_ai_scores[score_key] = raw_ai_scores_dict[found_key]
                else:
                    st.warning(f"AI did not return data for '{uni_name}'. Scores will be 0.")
                    normalized_ai_scores[score_key] = {
                        f['id']: {"score": 0, "note": "Data not found"} for f in AI_SCORED_FACTORS
                    }
                    normalized_ai_scores[score_key]['estimated_annual_cost'] = "0"
                    normalized_ai_scores[score_key]['estimated_starting_salary'] = "0"
                
                # Add a dummy 'opt' score if it wasn't requested, to prevent errors
                if 'opt' not in normalized_ai_scores[score_key]:
                     normalized_ai_scores[score_key]['opt'] = {"score": 0, "note": "N/A"}

            st.session_state.ai_scores = normalized_ai_scores

            # --- Perform Calculations ---
            scores_data = []
            table_data = []
            chart_data = [] 
            
            # --- NEW: Build active_weights dict for calculation ---
            active_weights = {}
            for factor in ALL_FACTORS:
                if factor['id'] == 'opt' and not st.session_state.opt_is_important:
                    continue
                active_weights[factor['id']] = st.session_state.weights[factor['id']]
            
            total_w = sum(active_weights.values()) or 1
            # --- END NEW ---
            
            for i, uni in enumerate(valid_universities):
                uni_name = uni['name']
                uni_major = uni['major']
                score_key = f"{uni_name}_{i}"
                
                weighted_score_sum = 0
                table_row_name = f"{uni_name} ({uni_major})"
                row = {"University": table_row_name}
                
                for factor in ALL_FACTORS:
                    fid = factor['id']
                    
                    if fid in [f['id'] for f in AI_SCORED_FACTORS]:
                        ai_score_obj = st.session_state.ai_scores.get(score_key, {}).get(fid, {"score": 0, "note": "N/A"})
                        score = ai_score_obj.get('score', 0)
                        note = ai_score_obj.get('note', 'N/A')
                        
                        row[factor['name']] = f"{score}/10 ({note})"
                    else:
                        score = st.session_state.user_scores.get(score_key, {}).get(fid, 0)
                        row[factor['name']] = f"{score}/10"
                    
                    # --- CALCULATION FIX: Only use active weights ---
                    if fid in active_weights:
                        weight_normalized = active_weights[fid] / total_w
                        contribution = score * weight_normalized
                        weighted_score_sum += contribution
                        
                        chart_data.append({
                            'University': table_row_name,
                            'Factor': factor['name'],
                            'Contribution': contribution * 10 # Scale to 100
                        })
                    # --- END CALCULATION FIX ---
                
                # --- ANNUAL COST & ROI Calculation ---
                cost_str = st.session_state.ai_scores.get(score_key, {}).get('estimated_annual_cost', '0')
                salary_str = st.session_state.ai_scores.get(score_key, {}).get('estimated_starting_salary', '0')
                
                cost_num = parse_dollars(cost_str)
                salary_num = parse_dollars(salary_str)
                
                row['Est. Annual Cost'] = f"${cost_num:,.0f}/yr"
                row['Est. Salary'] = f"${salary_num:,.0f}/yr"
                
                cost_note = st.session_state.ai_scores.get(score_key, {}).get('cost', {}).get('note', '0')
                net_cost_num = parse_dollars(cost_note)
                
                if net_cost_num == 0:
                    net_cost_num = cost_num - st.session_state.scholarships.get(score_key, 0)
                
                roi_percent = (salary_num / net_cost_num) * 100 if net_cost_num > 0 else 0
                row['Estimated ROI'] = f"{roi_percent:.1f}% (Salary / Net Cost)"
                # --- END ROI Calculation ---
                
                final_score = round(weighted_score_sum * 10, 1) # Scale to 1-100
                scores_data.append({'name': table_row_name, 'score': final_score})
                row['Final Score'] = final_score
                table_data.append(row)
            
            winner = max(scores_data, key=lambda x: x['score'])
            
            df_chart = pd.DataFrame(chart_data)
            
            st.session_state.calculations = {
                'scores': scores_data,
                'table': table_data, 
                'winner': winner,
                'df_chart': df_chart
            }
            
            st.session_state.messages = []
            st.success("Analysis Complete!")
        else:
            st.error("Failed to get AI scores. Please check the error messages.")

# --- Main Page (Results) ---
st.title("🎓 AI-Powered University Decision Matrix")
st.write("Compare universities by weighting what matters to you. Let AI find the objective data.")

if not st.session_state.calculations:
    st.info("Fill in the details on the left and click 'Generate AI Rankings' to see your results.")
    st.image("https.placehold.co/1200x600/FAFAFA/CCCCCC?text=Your+Results+Will+Appear+Here", use_container_width=True)
else:
    calc = st.session_state.calculations
    
    # 1. Recommendation
    st.success(f"**Recommendation: {calc['winner']['name']}**")
    st.markdown(f"Based on your weights, **{calc['winner']['name']}** is the best fit with a score of **{calc['winner']['score']}**.")

    # 2. Stacked Bar Chart
    st.subheader("Weighted Score Breakdown")
    
    df_chart = calc['df_chart']
    fig = px.bar(
        df_chart, 
        x='University', 
        y='Contribution', 
        color='Factor', 
        title='How Each Factor Contributes to the Final Score',
        hover_data=['Contribution']
    )
    fig.update_layout(
        yaxis_title="Final Score (out of 100)",
        xaxis_title="University",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Detailed Table
    st.subheader("Detailed Score Breakdown")
    
    df = pd.DataFrame(calc['table'])
    df = df.set_index("University")
    
    # --- DYNAMIC TABLE COLUMNS ---
    # Build the list of columns based on user's choice
    factor_cols = []
    for f in ALL_FACTORS:
        if f['id'] == 'opt' and not st.session_state.opt_is_important:
            continue
        factor_cols.append(f['name'])
    # --- END DYNAMIC COLUMNS ---
    
    data_cols = ['Est. Annual Cost', 'Est. Salary', 'Estimated ROI', 'Final Score']
    all_cols = factor_cols + [c for c in data_cols if c in df.columns]
    df = df[all_cols]
    
    def style_ai_columns(col_name):
        is_ai_col = any(col_name == f['name'] for f in AI_SCORED_FACTORS)
        return 'background-color: #EFF6FF' if is_ai_col else None
        
    st.dataframe(
        df.style.map_index(style_ai_columns, axis=1)
                .apply(lambda x: ['background-color: #DBEAFE' if x.name in data_cols else '' for i in x], axis=0)
                .format("{:.1f}", subset=['Final Score'])
    )
    st.caption("Blue-tinted columns are scored by AI. White columns are your manual scores. Grey columns are calculated data.")
    
    # --- Chat Interface ---
    st.divider()
    st.subheader("Ask About Your Results")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("e.g., How much was the total cost for Harvard?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                results_context = json.dumps(calc['table'])
                response = fetch_chat_response(st.session_state.messages, results_context)
                
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
