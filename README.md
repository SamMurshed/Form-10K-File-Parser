# Form-10K-File-Parser
Modular NLP pipeline to download, preprocess, and classify SEC 10-K filings. Combines TF-IDF + logistic regression and fine-tuned BERT with GPT-3.5-assisted labeling to extract financial highlights, risks, products, and AI mentions. Open-source, scalable, and designed for transparent financial text analysis.

This README document explains the workflow of the SEC 10-K NLP pipeline, including how the program works, required edits, and commands to run each step.

1. Download 10-K Filings
    What it does:
    Downloads raw 10-K HTML filings directly from the SEC EDGAR database and saves them locally in the data/ folder.
    File: notebooks/download_10k.py
    What to edit:
    Edit the companies list near the top of the file. For each company, specify:
    {
        "name": "Company Name",
        "url": "https://www.sec.gov/Archives/edgar/data/XXX/filing.htm",
        "save_path": "data/company_10k.html"
    }
    Run command:
    python notebooks/download_10k.py

2. Label Paragraphs with GPT-3.5
    What it does:
    Automatically labels paragraphs in downloaded 10-Ks using OpenAI’s GPT-3.5, categorizing them into financial highlights, risks, products, AI mentions, or other.
    File: notebooks/ai_label.py
    What to edit:
    Update the files list at the top of the file to include paths to downloaded 10-K HTMLs, e.g. "data/company_10k.html".
    
    Run command:
    python notebooks/ai_label.py
    Output:
    Labeled JSON files saved in ai_labeled/.

3. Train Logistic Regression Classifier
    What it does:
    Trains a TF-IDF vectorizer and logistic regression model on labeled data.
    File: notebooks/train_logistics.py
    What to edit:
    Ensure your combined manual and GPT-labeled data is saved as data/labeled_data.json.
    
    Run command:
    python notebooks/train_logistics.py
    Output:
    Model and vectorizer saved in models/logistic_model.pkl and models/vectorizer.pkl.

4. Batch Inference (Logistic + BERT)
    What it does:
    Runs both logistic regression and fine-tuned BERT models on paragraphs from a 10-K file.
    File: notebooks/batch_predict.py
    What to edit:
    Change the filepath variable to point to the desired 10-K HTML in data/.
    
    Run command:
    python notebooks/batch_predict.py
    Output:
    Predictions saved as JSON in predictions/.

5. Logistic-Only Prediction & Evaluation
    What it does:
    Runs logistic regression predictions on a 10-K and evaluates against ground truth if available.
    File: notebooks/predict_logistics.py
    What to edit:
    Change new_10k_path to your target 10-K HTML file.
    
    Run command:
    python notebooks/predict_logistics.py
    Output:
    Predictions saved in predicted_labels/, evaluation metrics printed if ground truth exists.

6. Evaluation of Predictions
    What it does:
    Compares predicted labels to ground truth labels and prints classification metrics.
    File: notebooks/calc_scores.py
    
    Run command:
    python notebooks/calc_scores.py

7. Interactive Paragraph Classification
    What it does:
    Allows manual input of paragraphs to classify in real-time using both logistic and BERT models.
    File: notebooks/predict.py
    
    Run command:
    python notebooks/predict.py
**Additional Notes
 - Make sure to have your OpenAI API key set in your environment for GPT labeling (ai_label.py).
 - Ensure dependencies (transformers, scikit-learn, torch, requests) are installed.
 - For BERT inference, GPU use is optional but recommended for speed.
 - Always keep your labeled data and models synchronized when retraining or evaluating.
