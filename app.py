import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the MarianMT model for English to Urdu
model_name = "Helsinki-NLP/opus-mt-en-ur"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Streamlit app title
st.title('English to Roman Urdu Translator')

# Text input for English prompt
english_text = st.text_area("Enter English Text:")

# Dictionary for Urdu to Roman Urdu conversion (extend this as needed)
urdu_to_roman_urdu = {
    "ہیلو": "salam",
    "آپ": "ap",
    "کیسے": "kese",
    "ہیں": "hain",
    "ہے": "hai",
    "ہوں": "hoon",
    "میں": "mein",
    "کرتا": "karta",
    "کرتی": "kartii",
    "ہوں": "hoon",
    # Add more mappings here
}

def urdu_to_roman(urdu_text):
    # Replace Urdu words with Roman Urdu equivalents
    roman_urdu_text = urdu_text
    for urdu_word, roman_word in urdu_to_roman_urdu.items():
        roman_urdu_text = roman_urdu_text.replace(urdu_word, roman_word)
    return roman_urdu_text

def translate_to_roman_urdu(english_text):
    # Tokenize the input text and translate
    translated = model.generate(**tokenizer(english_text, return_tensors="pt", padding=True))
    # Decode the output from Urdu script
    urdu_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    
    # Convert Urdu script to Roman Urdu
    roman_urdu_text = urdu_to_roman(urdu_text)
    
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
