import json
import os
import re

def save_individual_articles():
    input_file = "wikipedia_1000_sample.json"
    output_dir = "selected_articles"
    num_articles = 10

    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run the previous download script first.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory: {output_dir}")

    # Load articles
    print(f"Loading articles from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # Select articles
    selected = articles[:num_articles]
    
    saved_files = []

    print(f"Saving {len(selected)} articles to individual files...")
    
    for i, article in enumerate(selected):
        title = article.get('title', f'Article_{i}')
        # Sanitize filename: remove special chars, spaces to underscores
        safe_filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        # Limit filename length
        safe_filename = safe_filename[:50]
        
        file_path = os.path.join(output_dir, f"{safe_filename}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\n")
            f.write(f"URL: {article.get('url', 'N/A')}\n")
            f.write(f"ID: {article.get('id', 'N/A')}\n")
            f.write("-" * 40 + "\n\n")
            f.write(article.get('text', ''))
        
        saved_files.append(file_path)
        print(f"Saved: {file_path}")

    print(f"\nSuccessfully saved {len(saved_files)} articles in '{output_dir}/'.")

if __name__ == "__main__":
    save_individual_articles()
