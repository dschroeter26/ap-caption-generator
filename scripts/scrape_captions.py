import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_dvids_captions(base_url, num_pages=5):
    captions = []
    for page in range(1, num_pages + 1):
        url = f"{base_url}&page={page}"  # Adjust URL based on website pagination
        print(f"Scraping page: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all elements containing the search results
        for item in soup.find_all('div', class_='search_results_info_inner_wrapper'):
            # Extract title from assetTitle class
            title_tag = item.find('h2', class_='assetTitle')
            title = title_tag.get_text(strip=True) if title_tag else None

            # Extract photographer from details class
            photographer_tag = item.find('p', class_='details').find('a')
            photographer = photographer_tag.get_text(strip=True) if photographer_tag else None
            
            # Extract date from details class
            date_tag = item.find('p', class_='details')
            date = date_tag.text.strip().split('|')[-1].strip() if date_tag else None

            # Extract description from description class
            caption_tag = item.find('p', class_='description')
            caption = caption_tag.get_text(strip=True) if caption_tag else None

            # Add extracted data to list
            if caption:
                captions.append({
                    'title': title,
                    'caption': caption,
                    'date': date,
                    'photographer': photographer
                })

        time.sleep(2)  # Respectful scraping: add delay between requests

    return captions

def save_captions_to_csv(captions, filename='./../data/captions.csv'):
    df = pd.DataFrame(captions)
    df.to_csv(filename, index=False)
    print(f"Captions saved to {filename}")

# Usage
base_url = 'https://www.dvidshub.net/search/?q=&filter%5Btype%5D=image&view=list&sort=publishdate'
captions = scrape_dvids_captions(base_url, num_pages=100)
save_captions_to_csv(captions)
