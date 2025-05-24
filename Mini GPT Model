import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, pipeline
import json
import os

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device set to use: {device}")

###########################
# GPT Block (Transformer Block)
###########################
class GPTBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super(GPTBlock, self).__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        h = self.heads
        head_dim = C // h
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, h, head_dim).transpose(1, 2)
        k = k.view(B, T, h, head_dim).transpose(1, 2)
        v = v.view(B, T, h, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T)).to(x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(attended)
        x = self.norm1(x)
        x = x + self.ffn_dropout(self.ffn(x))
        x = self.norm2(x)
        return x

###########################
# GPT Model
###########################
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, heads, dropout=0.1, max_seq_len=512):
        super(GPT, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.blocks = nn.ModuleList([GPTBlock(embed_dim, heads, dropout) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_embed(x) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits

###########################
# Dataset (Hindi + Programming Text)
###########################
class TextDataset(Dataset):
    def __init__(self, text_file, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"डेटासेट फ़ाइल {text_file} नहीं मिली।")
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        self.tokens = self.tokenizer.encode(text)  # Directly use list of token IDs
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx+self.seq_length]
        target_ids = self.tokens[idx+1:idx+self.seq_length+1]
        return input_ids, target_ids

###########################
# Mood Detection Function (Multilingual)
###########################
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Failed to load sentiment analyzer: {e}. Falling back to neutral mood.")
    def detect_mood(text):
        return "neutral"
else:
    def detect_mood(text):
        try:
            result = sentiment_analyzer(text)[0]
            if result['score'] > 0.6 and 'positive' in result['label']:
                return "positive"
            elif result['score'] > 0.6 and 'negative' in result['label']:
                return "negative"
            else:
                return "neutral"
        except:
            return "neutral"

###########################
# Web Scraping Function
###########################
def web_search_scrape(query):
    print("Searching web for:", query)
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        results = soup.find_all('a', {'class': 'result__a'}, limit=1)
        if results:
            link = results[0]['href']
            print("Scraping:", link)
            page = requests.get(link, headers=headers, timeout=10)
            page.raise_for_status()
            page_soup = BeautifulSoup(page.text, 'html.parser')
            paragraphs = page_soup.find_all('p')
            text = " ".join([p.get_text() for p in paragraphs[:3]])
            return text or "मुझे उस विषय पर जानकारी नहीं मिली।"
        return "मुझे उस विषय पर जानकारी नहीं मिली।"
    except requests.RequestException as e:
        return f"वेब से जानकारी प्राप्त करने में त्रुटि: {str(e)}"

###########################
# Training Function
###########################
def train_model(model, train_loader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

###########################
# Text Generation with Context
###########################
def generate_text_with_context(model, tokenizer, conversation_history, max_length=100, device="cpu", temperature=0.7, top_k=50):
    prompt = " ".join(conversation_history[-5:])
    mood = detect_mood(conversation_history[-1] if conversation_history else "")
    if mood == "positive":
        prompt = "[Excited] " + prompt
    elif mood == "negative":
        prompt = "[Calm] " + prompt
    else:
        prompt = "[Neutral] " + prompt
    tokens = tokenizer.encode(prompt)  # Directly use list of token IDs
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    for _ in range(max_length):
        logits = model(tokens)[:, -1, :]
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        next_token = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices.gather(-1, next_token)
        tokens = torch.cat([tokens, next_token], dim=1)
    generated_text = tokenizer.decode(tokens[0].cpu().numpy(), skip_special_tokens=True)
    return generated_text[len(prompt):].strip(), mood

###########################
# Save and Load Conversation History
###########################
def save_history(history, file_path="/content/chat_history.json"):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

def load_history(file_path="/content/chat_history.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

###########################
# Chatbot Main Loop
###########################
def chatbot_loop(model, tokenizer, device):
    conversation_history = load_history()
    print("Chatbot शुरू हुआ। बातचीत बंद करने के लिए 'exit' टाइप करें।")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            save_history(conversation_history)
            print("चैटबॉट से विदा ले रहे हैं...")
            break
        conversation_history.append(user_input)
        conversation_history = conversation_history[-10:]  # Limit to last 10 messages
        if any(kw in user_input.lower() for kw in ["कोड", "प्रोग्राम", "पायथन", "लूप", "फंक्शन"]):
            prompt_prefix = "[Programming]"
        else:
            prompt_prefix = "[Hindi]"
        conversation_history[-1] = f"{prompt_prefix} {user_input}"
        response, mood = generate_text_with_context(model, tokenizer, conversation_history, max_length=100, device=device)
        if not response.strip() or "मुझे पता नहीं" in response.lower():
            web_info = web_search_scrape(user_input)
            response = f"वेब से जानकारी: {web_info}"
        conversation_history.append(response)
        save_history(conversation_history)
        print(f"Chatbot: {response} (Mood: {mood})")

###########################
# Main Execution
###########################
if __name__ == "__main__":
    # Install dependencies
    try:
        import torch, transformers, requests, bs4, sentencepiece
    except ImportError:
        print("Installing dependencies...")
        !pip install torch transformers requests beautifulsoup4 sentencepiece

    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        raise

    # Parameters
    vocab_size = tokenizer.vocab_size
    embed_dim = 256
    num_layers = 6
    heads = 8
    seq_length = 64
    batch_size = 16
    epochs = 5

    # Dataset path
    data_path = "/content/hindi_programming_text.txt"  # Update if using Google Drive

    # Mount Google Drive (optional)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        data_path = "/content/drive/MyDrive/hindi_programming_text.txt"
    except ImportError:
        print("Running locally or without Drive.")

    # Prepare dataset and dataloader
    dataset = TextDataset(data_path, tokenizer, seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GPT(vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers, heads=heads, dropout=0.1)

    # Train model
    train_model(model, train_loader, epochs, device)

    # Save model weights
    torch.save(model.state_dict(), "/content/model_weights.pth")

    # Start chatbot loop
    chatbot_loop(model, tokenizer, device)
