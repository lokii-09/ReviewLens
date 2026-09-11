import pandas as pd
from google_play_scraper import reviews, Sort
import time
import os

# 1. Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# 2. The official Play Store Package IDs for our target apps
app_packages = {
    'ChatGPT': 'com.openai.chatgpt',
    'Gemini': 'com.google.android.apps.bard',
    'Claude': 'com.anthropic.claude',
    'Perplexity': 'ai.perplexity.app.android',
    'Copilot': 'com.microsoft.copilot'
}

all_reviews = []
TARGET_REVIEWS_PER_APP = 1500

print("Starting the scrape. This might take a few minutes...")

for app_name, package_id in app_packages.items():
    print(f"Scraping {app_name} ({package_id})...")
    app_reviews = []
    continuation_token = None
    
    while len(app_reviews) < TARGET_REVIEWS_PER_APP:
        # Fetch a batch of reviews
        batch, continuation_token = reviews(
            package_id,
            lang='en', # English reviews
            country='us', # US store
            sort=Sort.NEWEST, # Get the most recent feedback
            count=200, # Number of reviews per batch
            continuation_token=continuation_token
        )
        
        if not batch:
            break # No more reviews available
            
        for r in batch:
            app_reviews.append({
                'app_name': app_name,
                'review_id': r['reviewId'],
                'content': r['content'],
                'score': r['score'],
                'thumbs_up': r['thumbsUpCount'],
                'date': r['at']
            })
            
        print(f"   Collected {len(app_reviews)} reviews for {app_name} so far...")
        
        # Polite delay to avoid rate limits and getting temporarily blocked
        time.sleep(1)
        
        # If we hit the end of the available reviews
        if not continuation_token:
            break

    # Keep only the target amount
    all_reviews.extend(app_reviews[:TARGET_REVIEWS_PER_APP])

# 3. Convert to DataFrame and save
df = pd.DataFrame(all_reviews)
df.to_csv('data/raw_reviews.csv', index=False)

print(f"\nSuccess! Total reviews scraped: {len(df)}")
print("Saved to data/raw_reviews.csv")