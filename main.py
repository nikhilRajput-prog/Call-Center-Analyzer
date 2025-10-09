import streamlit as st
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import plotly.express as px

load_dotenv()

st.set_page_config(
    page_title="CallAnalyzer AI",   
    page_icon=":telephone_receiver:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None
if 'audio_url' not in st.session_state:
    st.session_state.audio_url = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'call_stages' not in st.session_state:
    st.session_state.call_stages = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E40AF;
        margin-bottom: 1rem;
        width: 100%;
        text-align: left;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def render_header():
    st.markdown("<h1 class='main-header'>CallAnalyzer AI</h1>", unsafe_allow_html=True)
    st.markdown("<p><strong>AI-Powered Call Center Intelligence</strong></p>", unsafe_allow_html=True)
    
    with st.expander("About CallAnalyzer AI"):
        st.markdown("""
        <div>
            <p><strong>CallAnalyzer AI</strong> is a tool for analyzing call center and sales calls using advanced AI. It provides:</p>
            <ul>
                <li>Accurate transcription of audio calls</li>
                <li>Segmentation of calls into key stages (e.g., intro, resolution)</li>
                <li>Sentiment analysis to gauge customer emotions</li>
                <li>Compliance and script adherence checking</li>
                <li>Performance metrics and coaching tips</li>
                <li>Exportable reports in PDF</li>
            </ul>
            <p>Powered by Mistral AI's Voxtral model for transcription and insights.</p>
        </div>
        """, unsafe_allow_html=True)

def render_api_config():
    with st.expander("API Configuration"):
        st.markdown("<p>Enter your <a href='https://console.mistral.ai/' target='_blank'>Mistral AI API key</a> below.</p>", unsafe_allow_html=True)
        api_key = st.text_input("Mistral API Key", type="password", placeholder="Enter your API key here", value=st.session_state.api_key)
        
        if api_key:
            st.session_state.api_key = api_key
            st.markdown("<div class='success-box'>API key saved! You can now analyze calls.</div>", unsafe_allow_html=True)
        elif os.getenv("MISTRAL_API_KEY"):
            st.markdown("<div class='info-box'>Using default API key from .env file. You can provide your own above.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warning-box'>No API key provided. Please enter a Mistral API key to use the app.</div>", unsafe_allow_html=True)

def transcribe_audio(audio_url=None, audio_file=None):
    """Transcribe audio using Mistral API with timestamps"""
    try:
        # Get API key from session state or .env
        api_key = st.session_state.api_key if st.session_state.api_key else os.getenv("MISTRAL_API_KEY")
        
        if not api_key:
            return None, "API key is required. Please provide a Mistral API key in the API Configuration section."
        
        url = "https://api.mistral.ai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        if audio_url:
            data = {
                'file_url': audio_url,
                'model': "voxtral-mini-2507",
                'timestamp_granularities': "segment"
            }
            response = requests.post(url, headers=headers, json=data)
        elif audio_file:
            files = {
                'file': audio_file,
                'model': (None, "voxtral-mini-2507"),
                'timestamp_granularities': (None, "segment")
            }
            response = requests.post(url, headers=headers, files=files)
        else:
            return None, "No audio provided"
        
        if response.status_code == 200:
            result = response.json()
            return result, None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Exception: {str(e)}"

def segment_call(segments):
    """Segment the call into different stages"""
    call_stages = {
        "intro": [],
        "issue_description": [],
        "resolution": [],
        "objection": [],
        "closing": [],
    }
    
    total_segments = len(segments)
    if total_segments > 0:
        intro_end = max(1, int(total_segments * 0.15))
        issue_end = max(intro_end, int(total_segments * 0.45))
        resolution_end = max(issue_end, int(total_segments * 0.65))   
        objection_end = max(resolution_end, int(total_segments * 0.85))

        call_stages["intro"] = segments[:intro_end]
        call_stages["issue_description"] = segments[intro_end:issue_end]    
        call_stages["resolution"] = segments[issue_end:resolution_end]
        call_stages["objection"] = segments[resolution_end:objection_end]
        call_stages["closing"] = segments[objection_end:]
    
    return call_stages

def calculate_talk_ratio(segments):
    """Calculate the talk ratio between agent and customer"""
    agent_duration = 0
    customer_duration = 0

    for i, segment in enumerate(segments):
        duration = segment.get("end", 0) - segment.get("start", 0)
        if i % 2 == 0:  # Agent segments (even indices)
            agent_duration += duration  
        else:  # Customer segments (odd indices)
            customer_duration += duration
    
    if customer_duration == 0:
        customer_duration = 1  # Avoid division by zero
        
    return agent_duration / customer_duration

def extract_call_metrics(transcript, segments):
    """Extract various metrics from the call"""
    word_count = len(transcript.split())
    duration = segments[-1]['end'] if segments else 0
    talk_ratio = calculate_talk_ratio(segments)

    filler_words = ["um", "uh", "like", "you know", "so"]
    filler_count = sum(transcript.lower().count(word) for word in filler_words)

    metrics = {
        "duration": duration,
        "word_count": word_count,
        "talk_ratio": talk_ratio,
        "filler_count": filler_count,
        "filler_frequency": filler_count / word_count if word_count > 0 else 0,
    }
    return metrics

def process_audio():
    """Process the uploaded audio or URL"""
    st.session_state.processing = True
    with st.spinner("Transcribing audio... This may take a moment."):
        if st.session_state.uploaded_audio:
            result, error = transcribe_audio(audio_file=st.session_state.uploaded_audio)
        elif st.session_state.audio_url:
            result, error = transcribe_audio(audio_url=st.session_state.audio_url)
        else:
            st.error("No audio provided")
            st.session_state.processing = False
            return
        
        if error:
            st.error(f"Transcription failed: {error}")
            st.session_state.processing = False
            return
        
        # Store results
        st.session_state.transcript = result.get("text", "")
        st.session_state.segments = result.get("segments", [])
        
        # Process the segments and calculate metrics
        st.session_state.call_stages = segment_call(st.session_state.segments)
        st.session_state.metrics = extract_call_metrics(st.session_state.transcript, st.session_state.segments)
        
    st.session_state.processing = False
    st.success("Call analysis complete!")

def render_audio_upload():
    st.markdown("<h2 class='sub-header'>Upload Call Recording</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='info-box'><strong>Upload Audio File</strong></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state.uploaded_audio = uploaded_file
            st.audio(uploaded_file, format='audio/mp3')
    
    with col2:
        st.markdown("<div class='info-box'><strong>Or Provide Audio URL</strong></div>", unsafe_allow_html=True)
        audio_url = st.text_input("Enter URL to MP3 audio", placeholder="https://example.com/call.mp3", label_visibility="collapsed")
        if audio_url:
            st.session_state.audio_url = audio_url
            st.markdown(f"Audio URL: {audio_url}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Analyze Call", use_container_width=True, disabled=st.session_state.processing):
            if st.session_state.uploaded_audio or st.session_state.audio_url:
                process_audio()
            else:
                st.error("Please upload an audio file or provide an audio URL")

def render_results_tabs():
    """Render results in tabs if transcript available"""
    if not st.session_state.transcript:
        return
    
    st.markdown("---")
    st.markdown("<h2 class='sub-header'>Call Analysis Results</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Transcript", "Overview"])
    
    with tabs[0]:
        # Transcript display
        st.markdown("### Full Transcript")
        st.text_area("Transcript", value=st.session_state.transcript, height=300, disabled=True)
        
        # Segments display
        if st.session_state.segments:
            st.markdown("### Transcript Segments")
            segments_data = []
            for segment in st.session_state.segments:
                segments_data.append({
                    "Start": f"{segment.get('start', 0):.1f}s",
                    "End": f"{segment.get('end', 0):.1f}s",
                    "Duration": f"{segment.get('end', 0) - segment.get('start', 0):.1f}s",
                    "Text": segment.get("text", "")
                })
            
            df = pd.DataFrame(segments_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Call stages display
        st.markdown("### Call Stages")
        cols = st.columns(5)
        stages = ["intro", "issue_description", "resolution", "objection", "closing"]
        for i, stage in enumerate(stages):
            with cols[i]:
                st.markdown(f"**{stage.capitalize().replace('_', ' ')}**")
                stage_text = " ".join([s["text"] for s in st.session_state.call_stages.get(stage, [])])
                if stage_text:
                    st.markdown(f"<div style='height:150px;overflow-y:auto;font-size:0.9em;'>{stage_text}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='height:150px;overflow-y:auto;font-size:0.9em;'>No content</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### Call Metrics")
        metrics = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{metrics.get('duration', 0):.1f}s")
        with col2:
            st.metric("Word Count", metrics.get('word_count', 0))
        with col3:
            st.metric("Talk Ratio (Agent:Customer)", f"{metrics.get('talk_ratio', 0):.1f}:1")
        with col4:
            st.metric("Filler Words", metrics.get('filler_count', 0))
        
        st.markdown("### Call Timeline")
        if st.session_state.segments:
            # Create a Gantt-style chart using bar chart instead of timeline
            timeline_data = []
            for i, segment in enumerate(st.session_state.segments):
                speaker = "Agent" if i % 2 == 0 else "Customer"
                # Find which stage this segment belongs to
                stage = "Unknown"
                for stage_name, stage_segments in st.session_state.call_stages.items():
                    if segment in stage_segments:
                        stage = stage_name
                        break
                
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                duration = end_time - start_time
                
                timeline_data.append({
                    "Speaker": speaker,
                    "Start": start_time,
                    "Duration": duration,
                    "Stage": stage.capitalize().replace('_', ' '),
                    "Text": segment.get("text", "")[:50] + "..." if len(segment.get("text", "")) > 50 else segment.get("text", ""),
                    "Segment": f"Segment {i+1}"
                })
            
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                
                # Create a horizontal bar chart to simulate timeline
                fig = px.bar(
                    df_timeline,
                    x="Duration",
                    y="Speaker",
                    color="Stage",
                    base="Start",
                    orientation='h',
                    hover_data=["Text", "Start", "Duration"],
                    title="Call Timeline by Speaker and Stage",
                    labels={"Duration": "Time (seconds)", "Start": "Start Time (s)"}
                )
                
                fig.update_layout(
                    height=300,
                    xaxis_title="Time (seconds)",
                    yaxis_title="Speaker",
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show a simple timeline table
                st.markdown("### Timeline Details")
                timeline_display = df_timeline[['Speaker', 'Stage', 'Start', 'Duration', 'Text']].copy()
                timeline_display['Start'] = timeline_display['Start'].round(1)
                timeline_display['Duration'] = timeline_display['Duration'].round(1)
                st.dataframe(timeline_display, use_container_width=True, hide_index=True)
            else:
                st.warning("No timeline data available")
        else:
            st.warning("No segments for timeline")

def main():
    render_header()
    render_api_config()
    st.markdown("<hr>", unsafe_allow_html=True)
    render_audio_upload()
    render_results_tabs()

if __name__ == "__main__":
    main()