# Author: Boxuan Shan + support from Google Gemini 3
import json
import os
import time
import argparse
from datasets import load_dataset

def download_wikipedia(limit, output_file, random_shuffle=False):
    dataset_name = "wikimedia/wikipedia"
    config = "20231101.en"
    
    print("="*60)
    print(f"DOWNLOADING WIKIPEDIA ARTICLES")
    print(f"Target Count: {limit}")
    print(f"Output File:  {os.path.abspath(output_file)}")
    print(f"Random Shuffle: {random_shuffle}")
    print("="*60)
    
    start_time = time.time()
    
    # Load dataset in streaming mode
    print(f"Initializing stream from {dataset_name} ({config})...")
    dataset = load_dataset(dataset_name, config, split='train', streaming=True)
    
    if random_shuffle:
        print("Shuffling stream (buffer_size=10,000)...")
        # Buffer size ensures local randomness
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
    
    count = 0
    
    # Determine output format based on extension
    is_jsonl = output_file.endswith('.jsonl')
    
    if is_jsonl:
        # JSONL Mode (Preferred for large datasets)
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in dataset:
                f.write(json.dumps(article) + '\n')
                count += 1
                if count % 1000 == 0:
                    print(f"Collected {count} articles...", end='\r')
                if count >= limit:
                    break
    else:
        # Standard JSON Mode (Loads all into memory first - risky for huge datasets)
        articles = []
        for article in dataset:
            articles.append(article)
            count += 1
            if count % 1000 == 0:
                print(f"Collected {count} articles...", end='\r')
            if count >= limit:
                break
        
        print("\nSaving to JSON...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    print(f"\n\nDone! Collected {count} articles in {total_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wikipedia articles from Hugging Face.")
    
    parser.add_argument(
        "--count", 
        type=int, 
        default=1000,
        help="Number of articles to download (default: 1000)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="wikipedia_downloads.json",
        help="Output filename (e.g., 'data.json' or 'data.jsonl')"
    )
    
    parser.add_argument(
        "--random", 
        action="store_true", 
        help="Shuffle the dataset to get random articles instead of the first N"
    )

    args = parser.parse_args()
    
    download_wikipedia(args.count, args.output, args.random)
