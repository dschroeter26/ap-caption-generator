# AP Caption Generator Project

The **AP Caption Generator Project** is a machine learning-based tool that generates AP-style captions for images based on provided metadata. This project is built using Python and leverages Hugging Face's Transformers library to train a language model to generate captions in the Associated Press (AP) style. The project is divided into several scripts and components, each handling a specific aspect of the process.

## Key Components

1. **Data Preparation and Scraping**:
   - The `scrape_captions.py` script scrapes image captions from the DVIDS website to collect a dataset of AP-style captions.
   - Captions, along with metadata such as the date and photographer, are saved into CSV files for training.

2. **Data Analysis**:
   - The `analyze_data.py` script provides data analysis and visualization to understand the distribution and frequency of words and terms used in the scraped captions. This helps in understanding the dataset better before feeding it into the model.

3. **Data Preprocessing**:
   - The `preprocess_data.py` script cleans and tokenizes the text data from the CSV files to prepare it for training.
   - It splits the data into training and validation sets and saves them into new CSV files.

4. **Model Training**:
   - The `train_model.py` script is used to train a GPT-2-based model using the preprocessed training data.
   - It employs the Hugging Face Transformers library and PyTorch to fine-tune a pre-trained GPT-2 model on the AP-style caption dataset.

5. **Model Evaluation**:
   - The `evaluate_model.py` script evaluates the performance of the trained model using metrics such as BLEU and ROUGE.
   - This script helps determine how well the model is generating captions that are stylistically similar to the AP format.

6. **Model Testing**:
   - The `test_model.py` script is used to generate captions for unseen images or text inputs.
   - It loads the trained model and tests its caption-generation capability on new data, demonstrating its practical application.

7. **Dependencies**:
   - The project relies on several Python libraries such as `transformers`, `datasets`, `pandas`, `scikit-learn`, and `matplotlib` for scraping, data processing, model training, evaluation, and testing.

## How to Use

1. **Install Dependencies**:
   - Use the `requirements.txt` file to install all necessary dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Scrape Captions**:
   - Run the `scrape_captions.py` script to collect AP-style captions from the web:
     ```bash
     python scrape_captions.py
     ```

3. **Analyze Data**:
   - Run the `analyze_data.py` script to visualize the dataset:
     ```bash
     python analyze_data.py
     ```

4. **Preprocess Data**:
   - Prepare the data for training by running:
     ```bash
     python prepare_data.py
     ```

5. **Train the Model**:
   - Train the model using the prepared dataset:
     ```bash
     python train_model.py
     ```

6. **Evaluate the Model**:
   - Evaluate the trained model’s performance:
     ```bash
     python evaluate_model.py
     ```

7. **Test the Model**:
   - Test the model with new data to generate captions:
     ```bash
     python test_model.py
     ```

## Next Steps
- Integrate this caption generator with a frontend application (such as a React Native app) to automate the entire process from image upload to caption generation.
- Fine-tune the model further to improve accuracy and handle more diverse scenarios.
