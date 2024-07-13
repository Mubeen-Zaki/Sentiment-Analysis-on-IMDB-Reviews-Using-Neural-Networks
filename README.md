```markdown
# Sentiment Analysis Flask App

This Flask web application analyzes sentiment of IMDb movie reviews using LSTM neural networks.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources (if not already downloaded):
   ```bash
   python -m nltk.downloader punkt
   ```
4. GloVE File can be downloaded from : https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt

## Running the App

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open a web browser and go to `http://localhost:5000` to view the app.

## Usage

- Enter an IMDb movie review and click "Submit" to see sentiment analysis results.
- The app predicts whether the review is positive or negative.

## Folder Structure

- `app.py`: Flask web application script.
- `templates/`: HTML templates for the web app.
- `static/`: Static files (e.g., CSS, images).

## Dependencies

See `requirements.txt` for a list of Python libraries required to run the app.

## License

This project is licensed under the MIT License.
```
