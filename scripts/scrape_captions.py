import os
import time  # Import the time module to introduce delays
import requests
from bs4 import BeautifulSoup
import csv

def generate_unique_filename(url):
    """
    Generate a unique filename from the image URL.
    """
    return url.split('/')[-2] + "_" + url.split('/')[-1]

def scrape_dvids_captions(base_url, num_pages=100, save_images=True, image_folder='./../data/images', delay=1):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Use a set to keep track of processed image URLs
    processed_images = set()

    all_captions = []

    for page in range(1, num_pages + 1):
        page_url = f"{base_url}&page={page}"
        print(f"Scraping page: {page_url}")
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all elements with the specific image and metadata container
        search_results = soup.select('.search_results_oldstyle li')

        for item in search_results:
            image_tag = item.select_one('img')
            title_tag = item.select_one('.assetTitle a')
            photographer_tag = item.select_one('.details a')
            date_tag = item.select_one('.details')

            # Extract the required data
            if image_tag and title_tag and photographer_tag and date_tag:
                image_url = image_tag['src']
                title = title_tag.text.strip()
                photographer = photographer_tag.text.strip() if photographer_tag else "Unknown"
                date = date_tag.text.split('|')[-1].strip()  # Extract date part from details

                # Check if this image URL has been processed already
                if image_url in processed_images:
                    continue  # Skip if already processed

                # Add to processed_images set
                processed_images.add(image_url)

                # Generate a unique filename using the image URL
                image_name = os.path.join(image_folder, generate_unique_filename(image_url))

                # Download and save the image if requested
                if save_images:
                    try:
                        with open(image_name, 'wb') as img_file:
                            img_file.write(requests.get(image_url).content)
                        print(f"Saved image: {image_name}")
                    except Exception as e:
                        print(f"Error saving image {image_url}: {e}")
                        continue  # Skip this entry if there's an error

                # Save the caption data
                caption = {
                    'title': title,
                    'caption': item.select_one('.description').text.strip(),
                    'date': date,
                    'photographer': photographer,
                    'image_url': image_url,
                    'image_path': image_name  # Save the image path
                }
                all_captions.append(caption)

        # Delay between requests
        time.sleep(delay)

    # Save to CSV
    csv_file = './../data/captions.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=caption.keys())
        writer.writeheader()
        writer.writerows(all_captions)

    print(f"Captions saved to {csv_file}")

# Usage
scrape_dvids_captions('https://www.dvidshub.net/search/?filter%5Btype%5D=image&view=list&sort=publishdate', delay=2)
