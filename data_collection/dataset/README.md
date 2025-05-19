
# Research Abstract Classification

## 📌 Project Summary
- This project classifies research abstracts into five categories.
- Categories: Deep Learning, Wireless Communication, LLM, Cloud Computing, Virtual Reality.
- Preprocessing includes text cleaning, stopword removal, and lemmatization.

## ⚙️ Development Environment
- Python 3.10+
- Libraries used: `pandas`, `nltk`, `matplotlib`, `seaborn`

## 🧪 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/research-abstract-classification.git
   cd research-abstract-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. Run the data processing script:
   ```bash
   python data_preprocessing.py
   ```

## 📁 Dataset
- Five datasets merged into one unified CSV file.
- Each class is balanced to exactly 5000 records.
- Processed text is stored in the `summary_processed` column.

## 📊 Dataset Visualizations
- Label distribution bar chart
- Text length histogram
- Missing value heatmap

*You will add graphs later using `matplotlib` and `seaborn`.*

## 💻 User Interface
*(To be added later)*  
Screenshot of the interface (classification input, output).

## 🔮 Next Steps
- Train/Test split  
- Model training (e.g., SVM, LSTM, BERT)  
- Evaluation metrics  
- Save/export trained model  
- Deploy model on a web interface or API  

## 👨‍💻 Author
- Your Name  
- Your University / Department  
- Course: [Course Title]
