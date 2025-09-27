import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import warnings
import io
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AskREB - AI Text Classification",
    page_icon="🦝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #1F5F8B;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LlamaTextClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False

    def load_model(self, hf_token: str, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """Load the Llama model"""
        try:
            with st.spinner("Loading Llama model... This may take a few minutes."):
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=hf_token,
                    trust_remote_code=True
                )
                
                # Ensure proper padding
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=hf_token,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                self.device = next(self.model.parameters()).device
                self.model_loaded = True
                
                return True, "Model loaded successfully!"
                
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def create_prompt(self, text: str, system_prompt: str, user_instruction: str) -> str:
        """Create proper Llama chat format prompt"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'{user_instruction}: "{text}"'}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def classify_text(self, text: str, system_prompt: str, user_instruction: str, max_tokens: int = 50) -> Dict[str, any]:
        """Classify text using custom prompt"""
        if not self.model_loaded:
            return {
                "text": text,
                "classification": "Error: Model not loaded",
                "raw_response": "Model not loaded",
                "status": "failed"
            }

        prompt = self.create_prompt(text, system_prompt, user_instruction)

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                return_attention_mask=True
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Extract generated text
            prompt_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][prompt_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            return {
                "text": text,
                "classification": response,
                "raw_response": response,
                "status": "success"
            }

        except Exception as e:
            return {
                "text": text,
                "classification": f"Error: {str(e)}",
                "raw_response": str(e),
                "status": "failed"
            }

    def classify_batch(self, texts: List[str], system_prompt: str, user_instruction: str, max_tokens: int = 50, progress_callback=None) -> List[Dict[str, any]]:
        """Classify multiple texts with progress tracking"""
        results = []

        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i + 1, len(texts))

            # Memory management
            if i > 0 and i % 10 == 0:
                torch.cuda.empty_cache()

            result = self.classify_text(text, system_prompt, user_instruction, max_tokens)
            results.append(result)

        return results

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = LlamaTextClassifier()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🦝 AskREB</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Raccoon Eating Bagel - AI Text Classification Tool</p>', unsafe_allow_html=True)

    # Sidebar for model configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # HuggingFace token input
        hf_token = st.text_input(
            "HuggingFace Token",
            type="password",
            help="Enter your HuggingFace token to access Llama models"
        )
        
        # Model selection
        model_options = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct"
        ]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Load model button
        if st.button("🚀 Load Model") and hf_token:
            success, message = st.session_state.classifier.load_model(hf_token, selected_model)
            if success:
                st.session_state.model_loaded = True
                st.success(message)
            else:
                st.error(message)
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✅ Model loaded and ready!")
        else:
            st.warning("⚠️ Please load a model first")

    # Main interface
    if not st.session_state.model_loaded:
        st.info("👈 Please configure and load a model in the sidebar to get started.")
        return

    # File upload
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and preview data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Shape: {df.shape}")
            
            # Column selection
            st.header("🎯 Select Text Column")
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_columns:
                st.error("No text columns found in your CSV file.")
                return
            
            selected_column = st.selectbox("Choose the column containing text to classify:", text_columns)
            
            # Preview data
            with st.expander("📊 Preview Data"):
                st.dataframe(df.head())
                st.write(f"**Selected column '{selected_column}' sample:**")
                st.write(df[selected_column].head().tolist())
            
            # Custom prompts
            st.header("✍️ Custom Classification Prompts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                system_prompt = st.text_area(
                    "System Prompt",
                    value="""You are an AI assistant that classifies text according to user instructions. Provide clear, concise responses based on the user's request.""",
                    height=150,
                    help="Define the AI's role and general behavior"
                )
            
            with col2:
                user_instruction = st.text_area(
                    "Classification Instruction",
                    value="Classify this text as positive or negative sentiment. Respond with only 'positive' or 'negative'",
                    height=150,
                    help="Specific instruction for how to classify each text"
                )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                max_tokens = st.slider("Maximum tokens for response", 5, 200, 50)
                sample_size = st.slider("Sample size (0 = all data)", 0, min(len(df), 1000), 0)
            
            # Test on sample
            st.header("🧪 Test Classification")
            if st.button("Test on First 3 Rows"):
                sample_texts = df[selected_column].head(3).tolist()
                
                with st.spinner("Testing classification..."):
                    test_results = st.session_state.classifier.classify_batch(
                        sample_texts, system_prompt, user_instruction, max_tokens
                    )
                
                st.subheader("Test Results:")
                for i, result in enumerate(test_results):
                    with st.expander(f"Row {i+1}: {result['text'][:50]}..."):
                        st.write(f"**Original Text:** {result['text']}")
                        st.write(f"**Classification:** {result['classification']}")
                        st.write(f"**Status:** {result['status']}")
            
            # Run classification
            st.header("🚀 Run Classification")
            if st.button("🔥 Classify All Data", type="primary"):
                # Prepare data
                texts_to_classify = df[selected_column].dropna().tolist()
                if sample_size > 0:
                    texts_to_classify = texts_to_classify[:sample_size]
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {current}/{total} texts...")
                
                # Run classification
                start_time = time.time()
                with st.spinner(f"Classifying {len(texts_to_classify)} texts..."):
                    results = st.session_state.classifier.classify_batch(
                        texts_to_classify, system_prompt, user_instruction, max_tokens, update_progress
                    )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Store results
                st.session_state.results = results
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Add results back to original dataframe
                df_with_results = df.copy()
                df_with_results['ai_classification'] = None
                df_with_results['ai_status'] = None
                
                for i, result in enumerate(results):
                    if i < len(df_with_results):
                        df_with_results.iloc[i, df_with_results.columns.get_loc('ai_classification')] = result['classification']
                        df_with_results.iloc[i, df_with_results.columns.get_loc('ai_status')] = result['status']
                
                # Display results
                st.success(f"✅ Classification complete! Processed {len(results)} texts in {processing_time:.2f} seconds")
                
                # Statistics
                successful = sum(1 for r in results if r['status'] == 'success')
                st.metric("Success Rate", f"{successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
                
                # Download results
                csv_buffer = io.StringIO()
                df_with_results.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_string,
                    file_name=f"askreb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Preview results
                with st.expander("📊 Results Preview"):
                    st.dataframe(df_with_results)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("**AskREB** - Built with ❤️ and 🦝 for AI text classification")

if __name__ == "__main__":
    main()
