import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the MarianMT model for English to Urdu (replace this with your fine-tuned model if available)
model_name = "Helsinki-NLP/opus-mt-en-ur"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Streamlit app title
st.title('English to Roman Urdu Translator')

# Text input for English prompt
english_text = st.text_area("Enter English Text:")

def translate_to_roman_urdu(english_text):
    # Tokenize the input text
    translated = model.generate(**tokenizer(english_text, return_tensors="pt", padding=True))
    # Decode the output
    urdu_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    # Simple Roman Urdu conversion logic (this is a placeholder; refine as needed)
    roman_urdu_text = urdu_text[0].replace("ہے", "hai").replace("کی", "ki")  # Simplified example
    
    return roman_urdu_text

if st.button("Translate"):
    if english_text:
        # Translate the English input to Roman Urdu
        roman_urdu_text = translate_to_roman_urdu(english_text)
        
        # Display the result
        st.write("Roman Urdu Translation:")
        st.success(roman_urdu_text)
    else:
        st.warning("Please enter some English text!")
