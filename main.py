import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import re
from tqdm import tqdm

# Set up argument parsing for command line options
parser = argparse.ArgumentParser(description='Process NYT metadata for AI-related articles sentiment analysis')
parser.add_argument('--test', action='store_true', help='Run on a small test set')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
parser.add_argument('--input_file', type=str, default='data/nyt-metadata.csv', help='Input CSV file path')
parser.add_argument('--output_file', type=str, default='data/nyt-ai-sentiment.csv', help='Output CSV file path')
parser.add_argument('--debug', action='store_true', help='Print additional debug information')
args = parser.parse_args()

# Define AI-related keywords with high precision
AI_TERMS = {
    'primary': [
        'artificial intelligence',
        'machine learning',
        'deep learning',
        'neural network',
        'neural networks'
    ],
    'context_required': [
        'ai'  # This needs additional context verification
    ],
    'context_words': [
        'technology', 'computer', 'algorithm', 'model', 'system', 
        'research', 'robot', 'automated', 'smart', 'intelligent'
    ]
}

# More sophisticated function to detect AI-related content
def is_ai_related_article(text, debug=False):
    """
    Determines if an article is about AI topics using multiple checks.
    """
    if text is None or pd.isna(text) or text.strip() == '':
        return False
        
    text = text.lower()
    
    # Check for primary terms that are definitive indicators
    for term in AI_TERMS['primary']:
        if term in text:
            if debug:
                print(f"Matched primary term: '{term}'")
            return True
    
    # Check for standalone "ai" with proper context
    ai_matches = re.finditer(r'\b(ai)\b', text)
    
    for match in ai_matches:
        # Extract a window of text around the match for context
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end]
        
        # Check if any context words are present
        for context_word in AI_TERMS['context_words']:
            if context_word in context:
                if debug:
                    print(f"Matched 'ai' with context: '{context_word}' in: {context}")
                return True
                
        # Additional checks for common AI phrasings
        if re.search(r'ai (system|model|algorithm|technology|research|program|tool)', context):
            if debug:
                print(f"Matched 'ai' with technical suffix in: {context}")
            return True
            
        if re.search(r'(use of|using|powered by|based on) ai', context):
            if debug:
                print(f"Matched 'ai' with usage pattern in: {context}")
            return True
    
    return False

# Custom dataset class for the text data
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

def main():
    print(f"Loading data from {args.input_file}...")
    # Load the dataset
    df = pd.read_csv(args.input_file)
    
    # Create a more comprehensive text field
    print("Creating comprehensive text field...")
    text_columns = ['abstract', 'lead_paragraph', 'snippet', 'headline']
    
    # Check which columns actually exist in the dataframe
    available_columns = [col for col in text_columns if col in df.columns]
    if not available_columns:
        print("Error: None of the expected text columns are present in the dataset")
        return
        
    # Combine available text columns
    df['text'] = ''
    for col in available_columns:
        df['text'] += ' ' + df[col].fillna('')
    
    df['text'] = df['text'].str.strip()
    
    # Filter for AI-related articles using improved detection
    print("Filtering for AI-related articles...")
    df['is_ai_related'] = df['text'].apply(lambda x: is_ai_related_article(x, args.debug))
    df_filtered = df[df['is_ai_related']].copy()
    
    print(f"Found {len(df_filtered)} AI-related articles out of {len(df)} total articles")
    
    # Print sample of matched articles
    if not df_filtered.empty:
        print("\nSample of matched AI-related articles:")
        for i, (idx, row) in enumerate(df_filtered.head(3).iterrows()):
            print(f"\nExample {i+1}:")
            print(f"Text preview: {row['text'][:150]}...")
            if args.debug:
                is_ai_related_article(row['text'], debug=True)
    else:
        print("No AI-related articles found! Please check the filtering criteria or data.")
        return
    
    # If test flag is set, only use a small subset
    if args.test:
        print("\nRunning in test mode with a small sample...")
        df_filtered = df_filtered.head(50)
    
    # Load the tokenizer and model for sentiment analysis
    print("\nLoading RoBERTa sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Create dataset and dataloader
    dataset = TextDataset(df_filtered['text'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    # Sentiment labels
    labels = ['negative', 'neutral', 'positive']
    
    # Process in batches
    print(f"Running sentiment analysis in batches of {args.batch_size}...")
    sentiments = []
    scores = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get the predicted class and its score
            pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
            pred_scores = predictions.max(dim=1).values.cpu().numpy()
            
            # Map numeric labels to text labels
            pred_labels_text = [labels[label] for label in pred_labels]
            
            sentiments.extend(pred_labels_text)
            scores.extend(pred_scores)
    
    # Add results to the dataframe
    df_filtered['sentiment'] = sentiments
    df_filtered['sentiment_score'] = scores
    
    # Save the processed dataframe
    print(f"Saving processed data to {args.output_file}...")
    # Keep only the necessary columns
    columns_to_keep = ['pub_date', 'text', 'sentiment', 'sentiment_score']
    df_filtered[columns_to_keep].to_csv(args.output_file, index=False)
    
    print("Processing complete!")
    
    # Print some statistics
    sentiment_counts = df_filtered['sentiment'].value_counts()
    print("\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(df_filtered)*100:.2f}%)")

if __name__ == "__main__":
    main()