
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
   git clone https://github.com/francky877832/MachineLearningClass.git
   cd dataset
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
   cd code
   ```

## 📁 Dataset
- Five datasets merged into one unified CSV file.
- Each class is balanced to exactly 5000 records.
- Processed text is stored in the `summary_processed` column.

## 📊 Dataset Visualizations
- Label distribution bar chart
- Text length histogram
- Missing value heatmap


## 💻 User Interface
![label-distribution](https://github.com/user-attachments/assets/42507955-27d7-4b92-a9f0-7d0e6a29ce19)

![text-length-distribution](https://github.com/user-attachments/assets/c20fbb08-3c80-41ca-ac06-3ca9ab250b04)

## 🔮 Next Steps
- Train/Test split  
- Model training (e.g., SVM, LSTM, BERT)  
- Evaluation metrics  
- Save/export trained model  
- Deploy model on a web interface or API  

## 👨‍💻 Author
- Francky Ronsard SAAH  
- Kocaeli University / Information System And Technology  
- Course: Introduction To Machine Learning
