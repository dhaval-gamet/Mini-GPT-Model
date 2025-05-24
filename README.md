# Mini-GPT-Model

यह प्रोजेक्ट एक छोटा GPT (Generative Pre-trained Transformer) मॉडल है, जिसे हिंदी और प्रोग्रामिंग भाषा के डाटा पर ट्रेन किया गया है। इसमें sentiment analysis, वेब scraping, और एक CLI चैटबॉट जैसी सुविधाएँ शामिल हैं।

## विशेषताएँ (Features)
- GPT आधारित Transformer architecture
- हिंदी और प्रोग्रामिंग text datasets पर ट्रेनिंग
- Sentiment (mood) detection
- वेब scraping द्वारा जानकारी प्राप्त करना
- Conversation history सेव और लोड करना
- CLI चैटबॉट इंटरफेस

## आवश्यकताएँ (Requirements)

- Python 3.8+
- torch
- transformers
- requests
- beautifulsoup4
- sentencepiece
- (Optional, Google Colab के लिए) google.colab

इंस्टॉल करने के लिए:
```bash
pip install torch transformers requests beautifulsoup4 sentencepiece
```

## Installation

1. इस repository को clone करें:
   ```bash
   git clone https://github.com/dhaval-gamet/Mini-GPT-Model.git
   cd Mini-GPT-Model
   ```

2. ऊपर दिए गए dependencies को install करें।

## Dataset

- ट्रेनिंग के लिए आपको एक text फाइल चाहिए जिसमें हिंदी और प्रोग्रामिंग से जुड़ा content हो।
- Default path: `/content/hindi_programming_text.txt`  
- Format: Plain text file (UTF-8 encoded)

## उपयोग कैसे करें (Usage)

1. स्क्रिप्ट को चलाएं:
   ```bash
   python "Mini GPT Model.py"
   ```
2. चैटबॉट शुरू होगा। आप CLI में हिंदी या प्रोग्रामिंग से जुड़े सवाल पूछ सकते हैं।
3. 'exit' टाइप करके चैटबॉट बंद करें।

## Model Training और Testing

- ट्रेनिंग पैरामीटर्स और dataset path को script के अंदर बदल सकते हैं।
- ट्रेनिंग के बाद मॉडल वेट्स `/content/model_weights.pth` में सेव हो जाते हैं।

## Web Scraping

- जब मॉडल को जवाब न आए, तब यह वेब से जानकारी लाने की कोशिश करता है।

## Sentiment Analysis (Mood Detection)

- मल्टीलिंगुअल sentiment-analyzer (`nlptown/bert-base-multilingual-uncased-sentiment`)

## फ़ाइल संरचना (File Structure)

- `Mini GPT Model.py` — मुख्य कोड (मॉडल, ट्रेनिंग, चैटबॉट आदि)
- `README.md` — यह फाइल
- (Optional) `requirements.txt` — dependencies की सूची

## योगदान (Contributing)

- Pull requests स्वागत हैं!
- कोई issue मिले तो Issue टैब में रिपोर्ट करें।

## License

MIT License

## संपर्क / Author

- Author: Dhaval Gamet
- [GitHub Profile](https://github.com/dhaval-gamet)
